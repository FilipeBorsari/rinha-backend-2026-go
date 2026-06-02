package handler

import (
	"io"
	"net/http"
	"strconv"
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
}

func New(store *vectorstore.VectorStore) *Handler {
	return &Handler{store: store}
}

func (h *Handler) Ready(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (h *Handler) FraudScore(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, 4096)

	bufPtr := bodyPool.Get().(*[]byte)
	buf := *bufPtr
	n, err := readBody(r.Body, buf)
	r.Body.Close()
	if err != nil {
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

	result := h.store.Score(query)

	// Build response on the stack — zero heap allocations.
	var arr [64]byte
	var k int
	if result.Approved {
		k = copy(arr[:], `{"approved":true,"fraud_score":`)
	} else {
		k = copy(arr[:], `{"approved":false,"fraud_score":`)
	}
	tail := strconv.AppendFloat(arr[k:k:len(arr)], float64(result.FraudScore), 'f', 1, 32)
	k += len(tail)
	arr[k] = '}'
	k++

	w.Header().Set("Content-Type", "application/json")
	w.Write(arr[:k]) //nolint:errcheck
}

func readBody(body io.Reader, buf []byte) (int, error) {
	n := 0
	for {
		if n == len(buf) {
			var extra [1]byte
			m, err := body.Read(extra[:])
			if m == 0 && err == io.EOF {
				return n, nil
			}
			if err == nil && m == 0 {
				continue
			}
			return 0, err
		}

		m, err := body.Read(buf[n:])
		n += m
		if err == nil {
			continue
		}
		if err == io.EOF {
			return n, nil
		}
		return 0, err
	}
}
