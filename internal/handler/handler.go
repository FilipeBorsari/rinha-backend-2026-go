package handler

import (
	"io"
	"net/http"
	"sync"

	"github.com/bytedance/sonic"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorize"
	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorstore"
)

var fraudResponses = [6][]byte{
	[]byte(`{"approved":true,"fraud_score":0.0}`),
	[]byte(`{"approved":true,"fraud_score":0.2}`),
	[]byte(`{"approved":true,"fraud_score":0.4}`),
	[]byte(`{"approved":false,"fraud_score":0.6}`),
	[]byte(`{"approved":false,"fraud_score":0.8}`),
	[]byte(`{"approved":false,"fraud_score":1.0}`),
}

var bodyPool = sync.Pool{
	New: func() any {
		b := make([]byte, 4096)
		return &b
	},
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
	select {
	case h.sem <- struct{}{}:
		defer func() { <-h.sem }()
	default:
		w.WriteHeader(http.StatusServiceUnavailable)
		return
	}

	if r.Context().Err() != nil {
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, 4096)

	bufPtr := bodyPool.Get().(*[]byte)
	buf := *bufPtr
	n := 0
	for n < len(buf) {
		m, err := r.Body.Read(buf[n:])
		n += m
		if err != nil {
			if err != io.EOF {
				bodyPool.Put(bufPtr)
				r.Body.Close()
				http.Error(w, `{"error":"body error"}`, http.StatusBadRequest)
				return
			}
			break
		}
	}
	r.Body.Close()

	var req vectorize.Request
	if err := sonic.Unmarshal(buf[:n], &req); err != nil {
		bodyPool.Put(bufPtr)
		http.Error(w, `{"error":"invalid json"}`, http.StatusBadRequest)
		return
	}
	bodyPool.Put(bufPtr)

	query := vectorize.Vectorize(&req, h.store.Norm(), h.store.MccRisk())
	result := h.store.Search(query)

	w.Header().Set("Content-Type", "application/json")
	w.Write(fraudResponses[result.FraudCount])
}