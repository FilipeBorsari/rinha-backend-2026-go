package main

import (
	"log"
	"net/http"
	_ "net/http/pprof"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/handler"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorstore"
)

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}

	return fallback
}
// altos logs aqui ainda, para debug. LEMBRAR DE REMOVER ANTES DE ENTREGAR. --- IGNORE ---
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

	go func() {
		log.Println(http.ListenAndServe("0.0.0.0:6060", nil))
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", h.Ready)
	mux.HandleFunc("POST /fraud-score", h.FraudScore)

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("servidor ouvindo na porta %s", port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("erro no servidor: %v", err)
	}
}