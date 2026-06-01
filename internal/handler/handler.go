package handler

import (
	"fmt"
	"io"
	"net/http"
	"sync"
	"github.com/bytedance/sonic"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorize"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorstore"
)

var bodyPool = sync.Pool{
	New: func() any { b := make([]byte, 4096); return &b },
}

type Handler struct {
	store *vectorstore.VectorStore
	sem   chan struct{}
}

func New(store *vectorstore.VectorStore, concurrency int) *Handler {
	return &Handler{
		store: store,
		sem:   make(chan struct{}, concurrency),
	}
}

func (h *Handler) Ready(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (h *Handler) FraudScore(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 4096)

	bufPtr := bodyPool.Get().(*[]byte)
	buf := *bufPtr
	n, err := io.ReadFull(r.Body, buf)
	r.Body.Close()
	if err != nil && err != io.ErrUnexpectedEOF {
		bodyPool.Put(bufPtr)
		http.Error(w, `{"error":"invalid body"}`, http.StatusBadRequest)
		return
	}

	var req vectorize.Request
	unmarshalErr := sonic.ConfigFastest.Unmarshal(buf[:n], &req)
	if unmarshalErr != nil {
		bodyPool.Put(bufPtr)
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}

	query := vectorize.Vectorize(&req, h.store.Norm(), h.store.MccRisk())
	bodyPool.Put(bufPtr) 

	select {
	case h.sem <- struct{}{}:
		defer func() { <-h.sem }()
	case <-r.Context().Done():
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	result := h.store.Score(query)

	approved := "false"
	if result.Approved {
		approved = "true"
	}

	w.Header().Set("Content-Type", "application/json")
	fmt.Fprintf(w, `{"approved":%s,"fraud_score":%.1f}`, approved, result.FraudScore)
}
