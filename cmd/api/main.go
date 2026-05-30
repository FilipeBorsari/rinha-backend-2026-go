package main

import (
	"encoding/json"
	"log"
	"net/http"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"time"

	"github.com/bytedance/sonic"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/handler"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/timing"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorize"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorstore"
)

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return fallback
}
func setMaxProcsFromEnv() {
	v := os.Getenv("GOMAXPROCS")
	if v == "" {
		return
	}

	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		log.Printf("GOMAXPROCS invalido (%q), mantendo valor padrao", v)
		return
	}

	runtime.GOMAXPROCS(n)
}

func main() {
	setMaxProcsFromEnv()

	if err := sonic.Pretouch(reflect.TypeOf(vectorize.Request{})); err != nil {
		log.Printf("sonic.Pretouch falhou: %v", err)
	}

	refsPath := getEnv("REFERENCES_PATH", "/app/resources/references.json.gz")
	mccPath := getEnv("MCC_RISK_PATH", "/app/resources/mcc_risk.json")
	normPath := getEnv("NORMALIZATION_PATH", "/app/resources/normalization.json")
	port := getEnv("PORT", "8080")

	concurrency := 100
	if v := os.Getenv("CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			concurrency = n
		}
	}
	log.Printf("concurrency limit: %d", concurrency)

	log.Printf("carregando dataset de referencia: %s", refsPath)
	store, err := vectorstore.Load(refsPath, mccPath, normPath)
	if err != nil {
		log.Fatalf("erro ao carregar dataset: %v", err)
	}

	log.Printf("dataset carregado: %d vetores", store.Len())

	h := handler.New(store, concurrency)

	debugPort := getEnv("DEBUG_PORT", "6060")
	go func() {
		debugMux := http.NewServeMux()
		debugMux.HandleFunc("GET /debug/timings", func(w http.ResponseWriter, _ *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(timing.Global.Snapshot())
		})
		log.Printf("debug server ouvindo na porta %s", debugPort)
		if err := http.ListenAndServe(":"+debugPort, debugMux); err != nil {
			log.Printf("debug server encerrado: %v", err)
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", h.Ready)
	mux.HandleFunc("POST /fraud-score", h.FraudScore)

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  2 * time.Second,
		WriteTimeout: 5 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("servidor ouvindo na porta %s", port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("erro no servidor: %v", err)
	}
}
