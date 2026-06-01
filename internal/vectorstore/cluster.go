package vectorstore

import (
	"math/rand"
	"runtime"
	"sync"
)

func (s *VectorStore) buildIndex() {
	n := s.n
	data := s.data

	rng := rand.New(rand.NewSource(42))

	l1Float := kmeansMiniBatch(data, n, L1Count, 5000, 15, rng)
	l1Quant := quantizeCentroids(l1Float)

	l1Assign := make([]int32, n)
	nw := runtime.NumCPU()
	if nw > 8 {
		nw = 8
	}
	assignParallel(data, n, l1Quant, l1Assign, nw)
	l1Groups := groupByCluster(l1Assign, n, L1Count)

	type l2Result struct {
		centroids [][Dims]uint8
		assign    []int32
	}
	l2Results := make([]l2Result, L1Count)

	sem := make(chan struct{}, 8)
	var wg sync.WaitGroup

	for l1i, group := range l1Groups {
		wg.Add(1)
		sem <- struct{}{}
		go func(l1i int, group []int) {
			defer func() { <-sem; wg.Done() }()

			rng2 := rand.New(rand.NewSource(int64(l1i) * 1_000_003))
			gn := len(group)

			subData := make([]uint8, gn*Dims)
			for li, ri := range group {
				copy(subData[li*Dims:li*Dims+Dims], data[ri*Dims:ri*Dims+Dims])
			}

			k := L2PerL1
			if gn < k {
				k = gn
			}
			l2Float := kmeansMiniBatch(subData, gn, k, 2000, 10, rng2)
			// Pad to L2PerL1 with zero centroids if the group was too small.
			for len(l2Float) < L2PerL1 {
				l2Float = append(l2Float, [Dims]float32{})
			}
			l2Quant := quantizeCentroids(l2Float)

			assign := make([]int32, gn)
			for li := 0; li < gn; li++ {
				assign[li] = int32(nearestQuantCentroid(subData[li*Dims:li*Dims+Dims], l2Quant))
			}

			l2Results[l1i] = l2Result{centroids: l2Quant, assign: assign}
		}(l1i, group)
	}
	wg.Wait()

	s.l1s = make([]L1Cluster, L1Count)
	newOrder := make([]int, 0, n)

	for l1i, group := range l1Groups {
		copy(s.l1s[l1i].centroid[:], l1Quant[l1i][:])

		res := l2Results[l1i]
		l2Groups := groupByCluster(res.assign, len(group), L2PerL1)

		for l2i := 0; l2i < L2PerL1; l2i++ {
			start := int32(len(newOrder))
			for _, li := range l2Groups[l2i] {
				newOrder = append(newOrder, group[li])
			}
			copy(s.l1s[l1i].l2s[l2i].centroid[:], res.centroids[l2i][:])
			s.l1s[l1i].l2s[l2i].start = start
			s.l1s[l1i].l2s[l2i].length = int32(len(newOrder)) - start
		}
	}

	newData := make([]uint8, n*Dims)
	newLabels := make([]bool, n)
	for newIdx, oldIdx := range newOrder {
		copy(newData[newIdx*Dims:newIdx*Dims+Dims], data[oldIdx*Dims:oldIdx*Dims+Dims])
		newLabels[newIdx] = s.labels[oldIdx]
	}
	s.data = newData
	s.labels = newLabels
}

func kmeansMiniBatch(data []uint8, n, k, batchSize, iters int, rng *rand.Rand) [][Dims]float32 {
	centroids := make([][Dims]float32, k)
	if n == 0 {
		return centroids
	}

	actual := k
	if n < k {
		actual = n
	}

	perm := rng.Perm(n)
	for ci := 0; ci < actual; ci++ {
		ri := perm[ci]
		off := ri * Dims
		for d := 0; d < Dims; d++ {
			if data[off+d] == SentinelU8 {
				centroids[ci][d] = -1
			} else {
				centroids[ci][d] = float32(data[off+d]) / quantScale
			}
		}
	}

	bs := batchSize
	if bs > n {
		bs = n
	}
	counts := make([]int, k)

	for iter := 0; iter < iters; iter++ {
		for s := 0; s < bs; s++ {
			ri := rng.Intn(n)
			off := ri * Dims
			c := nearestFloat32Centroid(data[off:off+Dims], centroids[:actual])
			counts[c]++
			lr := float32(1.0) / float32(counts[c])
			for d := 0; d < Dims; d++ {
				if data[off+d] == SentinelU8 {
					continue
				}
				v := float32(data[off+d]) / quantScale
				if centroids[c][d] < 0 {
					centroids[c][d] = v
				} else {
					centroids[c][d] += lr * (v - centroids[c][d])
				}
			}
		}
	}

	return centroids
}

func nearestFloat32Centroid(vec []uint8, centroids [][Dims]float32) int {
	best := 0
	var bestDist float32 = 1e18
	for ci, c := range centroids {
		var dist float32
		for d := 0; d < Dims; d++ {
			if vec[d] == SentinelU8 || c[d] < 0 {
				continue
			}
			diff := float32(vec[d])/quantScale - c[d]
			if diff < 0 {
				diff = -diff
			}
			dist += diff
		}
		if dist < bestDist {
			bestDist = dist
			best = ci
		}
	}
	return best
}

func nearestQuantCentroid(vec []uint8, centroids [][Dims]uint8) int {
	best := 0
	var bestDist uint32 = ^uint32(0)
	for ci, c := range centroids {
		var dist uint32
		for d := 0; d < Dims; d++ {
			ai, bi := vec[d], c[d]
			if ai == SentinelU8 || bi == SentinelU8 {
				continue
			}
			if ai >= bi {
				dist += uint32(ai - bi)
			} else {
				dist += uint32(bi - ai)
			}
		}
		if dist < bestDist {
			bestDist = dist
			best = ci
		}
	}
	return best
}

func quantizeCentroids(centroids [][Dims]float32) [][Dims]uint8 {
	out := make([][Dims]uint8, len(centroids))
	for ci, c := range centroids {
		for d := 0; d < Dims; d++ {
			if c[d] < 0 {
				out[ci][d] = SentinelU8
			} else {
				out[ci][d] = quantize(c[d])
			}
		}
	}
	return out
}

func assignParallel(data []uint8, n int, centroids [][Dims]uint8, assign []int32, nWorkers int) {
	chunkSize := (n + nWorkers - 1) / nWorkers
	var wg sync.WaitGroup
	for w := 0; w < nWorkers; w++ {
		start := w * chunkSize
		end := start + chunkSize
		if end > n {
			end = n
		}
		if start >= n {
			break
		}
		wg.Add(1)
		go func(start, end int) {
			defer wg.Done()
			for i := start; i < end; i++ {
				off := i * Dims
				assign[i] = int32(nearestQuantCentroid(data[off:off+Dims], centroids))
			}
		}(start, end)
	}
	wg.Wait()
}

func groupByCluster(assign []int32, n, k int) [][]int {
	groups := make([][]int, k)
	for i := 0; i < n; i++ {
		c := int(assign[i])
		if c >= 0 && c < k {
			groups[c] = append(groups[c], i)
		}
	}
	return groups
}
