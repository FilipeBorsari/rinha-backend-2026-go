package vectorstore

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"
)

const (
	Dims = 14
	K    = 5

	SentinelU8 = 255
	quantScale  = 250.0
)

type VectorStore struct {
	data         []uint8
	labels       []bool
	n            int
	mccRisk      map[string]float32
	normConsts   NormConstants
	partStart    [9]int32
	partClusters [8][]subCluster
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

func (s *VectorStore) Len() int                        { return s.n }
func (s *VectorStore) MccRisk() map[string]float32     { return s.mccRisk }
func (s *VectorStore) Norm() NormConstants             { return s.normConsts }

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

	partStart := buildPartitions(data, labels, len(labels))

	log.Printf("rodando K-Means em %d particoes...", 8)
	t0 := time.Now()
	partClusters := buildAllClusters(data, labels, partStart)
	log.Printf("K-Means concluido em %s", time.Since(t0).Round(time.Millisecond))

	return &VectorStore{
		data:         data,
		labels:       labels,
		n:            len(labels),
		mccRisk:      mcc,
		normConsts:   norm,
		partStart:    partStart,
		partClusters: partClusters,
	}, nil
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

func vecPartKey(vec []uint8) int {
	key := 0
	if vec[9] > 0 {
		key |= 4
	}
	if vec[10] > 0 {
		key |= 2
	}
	if vec[11] > 0 {
		key |= 1
	}
	return key
}

func buildPartitions(data []uint8, labels []bool, n int) [9]int32 {
	var counts [8]int32
	for i := 0; i < n; i++ {
		counts[vecPartKey(data[i*Dims:i*Dims+Dims])]++
	}

	var starts [9]int32
	for k := 0; k < 8; k++ {
		starts[k+1] = starts[k] + counts[k]
	}

	newData := make([]uint8, len(data))
	newLabels := make([]bool, n)
	var offsets [8]int32
	copy(offsets[:], starts[:8])

	for i := 0; i < n; i++ {
		key := vecPartKey(data[i*Dims : i*Dims+Dims])
		pos := int(offsets[key])
		copy(newData[pos*Dims:(pos+1)*Dims], data[i*Dims:(i+1)*Dims])
		newLabels[pos] = labels[i]
		offsets[key]++
	}

	copy(data, newData)
	copy(labels, newLabels)
	return starts
}