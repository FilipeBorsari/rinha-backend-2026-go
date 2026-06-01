package vectorstore

const (
	L1Count  = 256
	L2PerL1  = 256
	L1Select = 12
	L2Select = 64

	fraudThreshold = 0.60
)

type KNNResult struct {
	FraudScore float32
	FraudCount int
	Approved   bool
}

type L2Cluster struct {
	centroid [Dims]uint8
	start    int32
	length   int32
}

type L1Cluster struct {
	centroid [Dims]uint8
	l2s      [L2PerL1]L2Cluster
}

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

type l2Ref struct {
	l1 int
	l2 int
}

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

	score := heap.fraudScore()
	return KNNResult{
		FraudScore: score,
		FraudCount: int(score * K),
		Approved:   score < fraudThreshold,
	}
}
