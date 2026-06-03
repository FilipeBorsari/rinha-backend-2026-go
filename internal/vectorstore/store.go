package vectorstore

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
)

const (
	Dims = 14
	K    = 11

	SentinelU8 = 255
	quantScale = 250.0
)

type VectorStore struct {
	data       []uint8
	labels     []bool
	n          int
	mccRisk    map[string]float32
	normConsts NormConstants
	l1s        []L1Cluster
}

type NormConstants struct {
	MaxAmount        float32 `json:"max_amount"`
	MaxInstallments  float32 `json:"max_installments"`
	AmountVsAvgRatio float32 `json:"amount_vs_avg_ratio"`
	MaxMinutes       float32 `json:"max_minutes"`
	MaxKm            float32 `json:"max_km"`
	MaxTxCount24h    float32 `json:"max_tx_count_24h"`
	MaxMerchantAvg   float32 `json:"max_merchant_avg_amount"`
}

func (s *VectorStore) Len() int                    { return s.n }
func (s *VectorStore) MccRisk() map[string]float32 { return s.mccRisk }
func (s *VectorStore) Norm() NormConstants         { return s.normConsts }

func (s *VectorStore) Vector(i int) []uint8 {
	off := i * Dims
	return s.data[off : off+Dims]
}

func (s *VectorStore) IsFraud(i int) bool {
	return s.labels[i]
}

func quantize(v float32) uint8 {
	if v < 0 {
		return SentinelU8
	}
	if v >= 1.0 {
		return 250
	}
	return uint8(v * quantScale)
}

func dequantize(b uint8) float32 {
	if b == SentinelU8 {
		return -1.0
	}
	return float32(b) / quantScale
}

func Load(refsPath, mccPath, normPath string) (*VectorStore, error) {
	norm, err := loadNorm(normPath)
	if err != nil {
		return nil, fmt.Errorf("normalization: %w", err)
	}

	mcc, err := loadMCC(mccPath)
	if err != nil {
		return nil, fmt.Errorf("mcc_risk: %w", err)
	}

	data, labels, err := loadRefs(refsPath)
	if err != nil {
		return nil, fmt.Errorf("references: %w", err)
	}

	s := &VectorStore{
		data:       data,
		labels:     labels,
		n:          len(labels),
		mccRisk:    mcc,
		normConsts: norm,
	}
	s.buildIndex()
	return s, nil
}

type refEntry struct {
	Vector []float64 `json:"vector"`
	Label  string    `json:"label"`
}

func loadRefs(path string) ([]uint8, []bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, err
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)

	if _, err := dec.Token(); err != nil {
		return nil, nil, err
	}

	data := make([]uint8, 0, 3_000_000*Dims)
	labels := make([]bool, 0, 3_000_000)

	var entry refEntry
	for dec.More() {
		if err := dec.Decode(&entry); err != nil {
			return nil, nil, err
		}
		if len(entry.Vector) != Dims {
			return nil, nil, fmt.Errorf("vetor com %d dimensões (esperado %d)", len(entry.Vector), Dims)
		}
		for j := 0; j < Dims; j++ {
			data = append(data, quantize(float32(entry.Vector[j])))
		}
		labels = append(labels, entry.Label == "fraud")
	}

	return data, labels, nil
}

func loadMCC(path string) (map[string]float32, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var raw map[string]float64
	if err := json.Unmarshal(b, &raw); err != nil {
		return nil, err
	}
	out := make(map[string]float32, len(raw))
	for k, v := range raw {
		out[k] = float32(v)
	}
	return out, nil
}

func loadNorm(path string) (NormConstants, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return NormConstants{}, err
	}
	var n NormConstants
	return n, json.Unmarshal(b, &n)
}
