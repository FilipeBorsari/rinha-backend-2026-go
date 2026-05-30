package vectorstore

import (
	"time"

	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/timing"
)

const (
	L1Count  = 256 // total L1 super-centroids
	L2PerL1  = 256 // L2 sub-clusters per L1 super-centroid
	L1Select = 12  // top-L1 super-centroids selected per query (was 8, prunes ~95% of the space)
	L2Select = 64  // top-L2 sub-clusters selected per query (was 32, ~2,880 records examined)

	fraudThreshold = 0.60
)

// KNNResult is returned by Score.
type KNNResult struct {
	FraudScore float32
	FraudCount int
	Approved   bool
}

// L2Cluster holds a quantised sub-cluster centroid and the contiguous
// range [start, start+length) in the reordered data array.
type L2Cluster struct {
	centroid [Dims]uint8
	start    int32
	length   int32
}

// L1Cluster holds a quantised super-centroid and its L2PerL1 sub-clusters.
type L1Cluster struct {
	centroid [Dims]uint8
	l2s      [L2PerL1]L2Cluster
}

// manhattanDist computes the Manhattan (L1) distance between two uint8 vectors.
// Dimensions where either operand equals SentinelU8 are treated as missing and skipped.
// The Go compiler auto-vectorises this loop when built with GOAMD64=v3 (AVX2).
func manhattanDist(a, b []uint8) uint32 {
	var sum uint32
	for i := 0; i < Dims; i++ {
		ai, bi := a[i], b[i]
		if ai == SentinelU8 || bi == SentinelU8 {
			continue
		}
		if ai >= bi {
			sum += uint32(ai - bi)
		} else {
			sum += uint32(bi - ai)
		}
	}
	return sum
}

// squaredEuclideanDist computes the squared Euclidean (L2²) distance.
// Used for the final exact K-NN scan to match the ground-truth labelling
// which was generated with k=5 Euclidean brute-force.
// Squared Euclidean preserves the same ordering as Euclidean without a sqrt.
func squaredEuclideanDist(a, b []uint8) uint32 {
	var sum uint32
	for i := 0; i < Dims; i++ {
		ai, bi := a[i], b[i]
		if ai == SentinelU8 || bi == SentinelU8 {
			continue
		}
		var d uint32
		if ai >= bi {
			d = uint32(ai - bi)
		} else {
			d = uint32(bi - ai)
		}
		sum += d * d
	}
	return sum
}

// scanL1 returns the L1Select indices of the closest L1 super-centroids to query.
// Implements the L1 Root Scan: scans all L1Count centroids and prunes 97% of the space.
func scanL1(query *[Dims]uint8, l1s []L1Cluster) [L1Select]int {
	var topDist [L1Select]uint32
	var topIdx [L1Select]int
	for i := range topDist {
		topDist[i] = ^uint32(0)
	}
	worstPos := 0
	for i := range l1s {
		d := manhattanDist(query[:], l1s[i].centroid[:])
		if d < topDist[worstPos] {
			topDist[worstPos] = d
			topIdx[worstPos] = i
			worstPos = 0
			for j := 1; j < L1Select; j++ {
				if topDist[j] > topDist[worstPos] {
					worstPos = j
				}
			}
		}
	}
	return topIdx
}

// l2Ref identifies one L2 sub-cluster by its L1 and L2 indices.
type l2Ref struct {
	l1 int
	l2 int
}

// scanL2 scans all L2PerL1 sub-clusters of each selected L1 (2,048 candidates total)
// and returns the L2Select closest sub-clusters.
// Implements the L2 Leaf Scan.
func scanL2(query *[Dims]uint8, l1s []L1Cluster, l1Idxs [L1Select]int) [L2Select]l2Ref {
	var topDist [L2Select]uint32
	var topRef [L2Select]l2Ref
	for i := range topDist {
		topDist[i] = ^uint32(0)
	}
	worstPos := 0
	for _, l1i := range l1Idxs {
		l1 := &l1s[l1i]
		for j := range l1.l2s {
			d := manhattanDist(query[:], l1.l2s[j].centroid[:])
			if d < topDist[worstPos] {
				topDist[worstPos] = d
				topRef[worstPos] = l2Ref{l1: l1i, l2: j}
				worstPos = 0
				for k := 1; k < L2Select; k++ {
					if topDist[k] > topDist[worstPos] {
						worstPos = k
					}
				}
			}
		}
	}
	return topRef
}

type topKHeap struct {
	dists    [K]uint32
	isfraud  [K]bool
	size     int
	worstPos int
}

func (h *topKHeap) push(dist uint32, fraud bool) {
	if h.size < K {
		h.dists[h.size] = dist
		h.isfraud[h.size] = fraud
		h.size++
		if h.size == K {
			h.worstPos = 0
			for i := 1; i < K; i++ {
				if h.dists[i] > h.dists[h.worstPos] {
					h.worstPos = i
				}
			}
		}
		return
	}
	if dist < h.dists[h.worstPos] {
		h.dists[h.worstPos] = dist
		h.isfraud[h.worstPos] = fraud
		h.worstPos = 0
		for i := 1; i < K; i++ {
			if h.dists[i] > h.dists[h.worstPos] {
				h.worstPos = i
			}
		}
	}
}

func (h *topKHeap) fraudScore() float32 {
	var fraudCount int
	for i := 0; i < h.size; i++ {
		if h.isfraud[i] {
			fraudCount++
		}
	}
	if h.size == 0 {
		return 0
	}
	return float32(fraudCount) / float32(h.size)
}

func (s *VectorStore) Score(query [Dims]float32) KNNResult {
	t0 := time.Now()

	var qv [Dims]uint8
	for i, v := range query {
		qv[i] = quantize(v)
	}

	l1Idxs := scanL1(&qv, s.l1s)

	l2Refs := scanL2(&qv, s.l1s, l1Idxs)

	var heap topKHeap
	for _, ref := range l2Refs {
		cl := &s.l1s[ref.l1].l2s[ref.l2]
		start := int(cl.start)
		end := start + int(cl.length)
		for i := start; i < end; i++ {
			d := squaredEuclideanDist(qv[:], s.data[i*Dims:(i+1)*Dims])
			heap.push(d, s.labels[i])
		}
	}

	timing.Global.Classify.Record(time.Since(t0).Nanoseconds())

	score := heap.fraudScore()
	return KNNResult{
		FraudScore: score,
		FraudCount: int(score * K),
		Approved:   score < fraudThreshold,
	}
}
