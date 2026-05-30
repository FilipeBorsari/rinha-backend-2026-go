package timing

import "sync/atomic"

type Bucket struct {
	TotalNs atomic.Int64
	Count   atomic.Int64
}

func (b *Bucket) Record(ns int64) {
	b.TotalNs.Add(ns)
	b.Count.Add(1)
}

type BucketSnapshot struct {
	Count   int64   `json:"count"`
	MeanUs  float64 `json:"mean_us"`
	TotalMs float64 `json:"total_ms"`
}

func (b *Bucket) Snapshot() BucketSnapshot {
	count := b.Count.Load()
	totalNs := b.TotalNs.Load()
	var meanUs float64
	if count > 0 {
		meanUs = float64(totalNs) / float64(count) / 1000.0
	}
	return BucketSnapshot{
		Count:   count,
		MeanUs:  meanUs,
		TotalMs: float64(totalNs) / 1e6,
	}
}

type Stats struct {
	ParseJSON Bucket
	Vectorize Bucket
	Classify  Bucket
}

type StatsSnapshot struct {
	ParseJSON BucketSnapshot `json:"parse_json"`
	Vectorize BucketSnapshot `json:"vectorize"`
	Classify  BucketSnapshot `json:"classify"`
}

func (s *Stats) Snapshot() StatsSnapshot {
	return StatsSnapshot{
		ParseJSON: s.ParseJSON.Snapshot(),
		Vectorize: s.Vectorize.Snapshot(),
		Classify:  s.Classify.Snapshot(),
	}
}

var Global Stats
