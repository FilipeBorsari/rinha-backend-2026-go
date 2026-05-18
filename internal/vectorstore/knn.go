package vectorstore

const (
	fraudThreshold = 0.6
	kNeighbors     = 5

	partitionPruneThreshold = uint32(62500)
)

type KNNResult struct {
	FraudScore float32
	Approved   bool
}

func queryPartKey(q [Dims]uint8) int {
	key := 0
	if q[9] > 0 {
		key |= 4
	}
	if q[10] > 0 {
		key |= 2
	}
	if q[11] > 0 {
		key |= 1
	}
	return key
}

func (s *VectorStore) Search(query [Dims]float32) KNNResult {
	var q [Dims]uint8
	for i := 0; i < Dims; i++ {
		q[i] = quantize(query[i])
	}

	var top5Dist [kNeighbors]uint32
	var top5Fraud [kNeighbors]bool
	count := 0
	maxIdx := 0
	threshold := ^uint32(0)

	partKey := queryPartKey(q)

	s.searchPartition(q, partKey, &top5Dist, &top5Fraud, &count, &maxIdx, &threshold)

	if count < kNeighbors || threshold >= partitionPruneThreshold {
		for pk := 0; pk < 8; pk++ {
			if pk == partKey {
				continue
			}
			s.searchPartition(q, pk, &top5Dist, &top5Fraud, &count, &maxIdx, &threshold)
		}
	}

	fraudCount := 0
	for _, isFraud := range top5Fraud[:count] {
		if isFraud {
			fraudCount++
		}
	}

	score := float32(fraudCount) / float32(kNeighbors)
	return KNNResult{
		FraudScore: score,
		Approved:   score < fraudThreshold,
	}
}

func (s *VectorStore) searchPartition(
	q [Dims]uint8, pk int,
	top5Dist *[kNeighbors]uint32, top5Fraud *[kNeighbors]bool,
	count *int, maxIdx *int, threshold *uint32,
) {
	clusters := s.partClusters[pk]
	if len(clusters) == 0 {
		for i, end := int(s.partStart[pk]), int(s.partStart[pk+1]); i < end; i++ {
			ref := s.data[i*Dims : i*Dims+Dims]
			updateTopK(q, ref, s.labels[i], top5Dist, top5Fraud, count, maxIdx, threshold)
		}
		return
	}

	topC := findTopClusters(q, clusters)

	for _, ci := range topC {
		if ci < 0 {
			break
		}
		cl := clusters[ci]
		for i := int(cl.start); i < int(cl.start+cl.size); i++ {
			ref := s.data[i*Dims : i*Dims+Dims]
			updateTopK(q, ref, s.labels[i], top5Dist, top5Fraud, count, maxIdx, threshold)
		}
	}
}

func updateTopK(
	q [Dims]uint8, ref []uint8, isfraud bool,
	top5Dist *[kNeighbors]uint32, top5Fraud *[kNeighbors]bool,
	count *int, maxIdx *int, threshold *uint32,
) {
	d := squaredDist(q, ref, *threshold)
	if *count < kNeighbors {
		top5Dist[*count] = d
		top5Fraud[*count] = isfraud
		*count++
		if *count == kNeighbors {
			*maxIdx = maxIndex(top5Dist[:])
			*threshold = top5Dist[*maxIdx]
		}
	} else if d < top5Dist[*maxIdx] {
		top5Dist[*maxIdx] = d
		top5Fraud[*maxIdx] = isfraud
		*maxIdx = maxIndex(top5Dist[:])
		*threshold = top5Dist[*maxIdx]
	}
}

func squaredDist(q [Dims]uint8, ref []uint8, threshold uint32) uint32 {
	var sum uint32
	for j := 0; j < Dims; j++ {
		qj := q[j]
		rj := ref[j]
		if qj == SentinelU8 || rj == SentinelU8 {
			continue
		}
		diff := int32(qj) - int32(rj)
		sum += uint32(diff * diff)
		if sum >= threshold {
			return sum
		}
	}
	return sum
}

func maxIndex(a []uint32) int {
	m := 0
	for i := 1; i < len(a); i++ {
		if a[i] > a[m] {
			m = i
		}
	}
	return m
}
