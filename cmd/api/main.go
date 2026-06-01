package main

import (
	"fmt"
	"net/http"
	"os"
	"reflect"
	"runtime"
	"strconv"
	"time"

	"github.com/bytedance/sonic"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/handler"
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
		return
	}

	runtime.GOMAXPROCS(n)
}

func main() {
	setMaxProcsFromEnv()

	_ = sonic.Pretouch(reflect.TypeOf(vectorize.Request{}))

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

	store, err := vectorstore.Load(refsPath, mccPath, normPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	h := handler.New(store, concurrency)

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

	if err := srv.ListenAndServe(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
