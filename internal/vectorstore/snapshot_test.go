package vectorstore

import (
	"compress/gzip"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestSnapshotRoundTrip(t *testing.T) {
	t.Parallel()

	store := &VectorStore{
		data:   []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1},
		labels: []bool{true, false},
		n:      2,
		mccRisk: map[string]float32{
			"1234": 0.7,
			"5678": 0.2,
		},
		normConsts: NormConstants{
			MaxAmount:        42,
			MaxInstallments:  12,
			AmountVsAvgRatio: 3.5,
			MaxMinutes:       90,
			MaxKm:            120,
			MaxTxCount24h:    18,
			MaxMerchantAvg:   250,
		},
		l1s: make([]L1Cluster, 2),
	}
	store.l1s[0].centroid = [Dims]uint8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	store.l1s[0].l2s[0] = L2Cluster{centroid: [Dims]uint8{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, start: 0, length: 1}
	store.l1s[1].centroid = [Dims]uint8{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}
	store.l1s[1].l2s[1] = L2Cluster{centroid: [Dims]uint8{4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4}, start: 1, length: 1}

	path := filepath.Join(t.TempDir(), "references.bin")
	if err := WriteSnapshot(path, store); err != nil {
		t.Fatalf("WriteSnapshot() error = %v", err)
	}

	got, err := loadSnapshot(path)
	if err != nil {
		t.Fatalf("loadSnapshot() error = %v", err)
	}

	if !reflect.DeepEqual(got, store) {
		t.Fatalf("snapshot mismatch\nwant: %#v\ngot: %#v", store, got)
	}
}

func TestLoadFallsBackWhenSnapshotMissing(t *testing.T) {
	t.Parallel()

	baseDir := t.TempDir()
	refsPath, mccPath, normPath := writeFixtureFiles(t, baseDir)

	store, err := Load(filepath.Join(baseDir, "missing.bin"), refsPath, mccPath, normPath)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}

	if store.Len() != 2 {
		t.Fatalf("store.Len() = %d, want 2", store.Len())
	}
	if len(store.l1s) != L1Count {
		t.Fatalf("len(store.l1s) = %d, want %d", len(store.l1s), L1Count)
	}
	if store.MccRisk()["5411"] != 0.8 {
		t.Fatalf("store.MccRisk()[5411] = %v, want 0.8", store.MccRisk()["5411"])
	}
}

func writeFixtureFiles(t *testing.T, dir string) (string, string, string) {
	t.Helper()

	refsPath := filepath.Join(dir, "references.json.gz")
	mccPath := filepath.Join(dir, "mcc_risk.json")
	normPath := filepath.Join(dir, "normalization.json")

	entries := []map[string]any{
		{
			"vector": []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5},
			"label":  "fraud",
		},
		{
			"vector": []float64{0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.2, 0.4, 0.6, 0.8, 0.3, 0.1, 0.9, 0.7},
			"label":  "regular",
		},
	}

	f, err := os.Create(refsPath)
	if err != nil {
		t.Fatalf("os.Create(%q) error = %v", refsPath, err)
	}
	gz := gzip.NewWriter(f)
	if err := json.NewEncoder(gz).Encode(entries); err != nil {
		_ = gz.Close()
		_ = f.Close()
		t.Fatalf("encode references error = %v", err)
	}
	if err := gz.Close(); err != nil {
		_ = f.Close()
		t.Fatalf("gzip close error = %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("file close error = %v", err)
	}

	mccJSON := []byte(`{"5411":0.8}`)
	if err := os.WriteFile(mccPath, mccJSON, 0o644); err != nil {
		t.Fatalf("os.WriteFile(%q) error = %v", mccPath, err)
	}

	normJSON := []byte(`{"max_amount":1000,"max_installments":12,"amount_vs_avg_ratio":4,"max_minutes":1440,"max_km":100,"max_tx_count_24h":20,"max_merchant_avg_amount":500}`)
	if err := os.WriteFile(normPath, normJSON, 0o644); err != nil {
		t.Fatalf("os.WriteFile(%q) error = %v", normPath, err)
	}

	return refsPath, mccPath, normPath
}
