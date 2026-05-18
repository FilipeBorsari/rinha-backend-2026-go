package vectorstore

import (
	"math/rand/v2"
	"sync"
)

const (
	// kSubClusters is the number of K-Means sub-clusters per binary partition.
	// 64 clusters × 8 partitions = 512 total clusters.
	// Each cluster holds ~375k/64 ≈ 5859 vectors on a 3M dataset.
	kSubClusters = 64

	// kSearchClusters is how many sub-clusters we probe per partition during search.
	// 6/64 = 9.4% of each partition → ~35k candidates per query (≈10x reduction).
	kSearchClusters = 6

	// kmeansIters is the number of mini-batch K-Means iterations.
	kmeansIters = 20

	// kmeansBatch is the number of random samples per mini-batch iteration.
	kmeansBatch = 20_000
)

// subCluster holds a quantized centroid and its vector range in the flat data array.
type subCluster struct {
	centroid [Dims]uint8
	start    int32 // absolute index into VectorStore.data / .labels
	size     int32 // number of vectors in this cluster
}

// buildAllClusters builds sub-clusters for all 8 partitions in parallel.
// Must be called after buildPartitions (data/labels already sorted by partition key).
func buildAllClusters(data []uint8, labels []bool, partStart [9]int32) [8][]subCluster {
	var result [8][]subCluster
	var wg sync.WaitGroup

	for pk := 0; pk < 8; pk++ {
		wg.Add(1)
		go func(pk int) {
			defer wg.Done()
			pStart := int(partStart[pk])
			pEnd := int(partStart[pk+1])
			// Each goroutine writes to its own non-overlapping slice of data/labels,
			// so no mutex is needed.
			result[pk] = buildPartitionClusters(data, labels, pStart, pEnd, int64(pk*31337+17))
		}(pk)
	}

	wg.Wait()
	return result
}

// buildPartitionClusters runs mini-batch K-Means on the partition [pStart, pEnd),
// reorders data/labels in-place by cluster assignment, and returns the descriptors.
func buildPartitionClusters(data []uint8, labels []bool, pStart, pEnd int, seed int64) []subCluster {
	n := pEnd - pStart
	if n == 0 {
		return nil
	}

	k := kSubClusters
	if k > n {
		k = n
	}
	if k == 1 {
		// Trivial: single cluster, no reorder needed.
		var c subCluster
		c.start = int32(pStart)
		c.size = int32(n)
		c.centroid = centroidOfRange(data, pStart, pEnd)
		return []subCluster{c}
	}

	rng := rand.New(rand.NewPCG(uint64(seed), uint64(n)))

	// --- Random initialisation (Forgy): pick k distinct random vectors ---
	perm := rng.Perm(n)
	centroids := make([][Dims]float32, k)
	for i := 0; i < k; i++ {
		off := (pStart + perm[i]) * Dims
		for d := 0; d < Dims; d++ {
			centroids[i][d] = float32(data[off+d])
		}
	}

	// --- Mini-batch K-Means ---
	for iter := 0; iter < kmeansIters; iter++ {
		batchSize := kmeansBatch
		if batchSize > n {
			batchSize = n
		}
		batch := rng.Perm(n)[:batchSize]

		counts := make([]int32, k)
		sums := make([][Dims]float64, k)

		for _, idx := range batch {
			off := (pStart + idx) * Dims
			c := nearestCentroidIdx(data[off:off+Dims], centroids)
			counts[c]++
			for d := 0; d < Dims; d++ {
				sums[c][d] += float64(data[off+d])
			}
		}

		// Update centroids; re-init empty ones with a random batch point.
		for c := 0; c < k; c++ {
			if counts[c] == 0 {
				// Re-init with a random point from the batch.
				idx := batch[rng.IntN(batchSize)]
				off := (pStart + idx) * Dims
				for d := 0; d < Dims; d++ {
					centroids[c][d] = float32(data[off+d])
				}
				continue
			}
			fc := float64(counts[c])
			for d := 0; d < Dims; d++ {
				centroids[c][d] = float32(sums[c][d] / fc)
			}
		}
	}

	// --- Final full-partition assignment pass ---
	assignments := make([]int32, n)
	counts := make([]int32, k)
	for i := 0; i < n; i++ {
		off := (pStart + i) * Dims
		c := nearestCentroidIdx(data[off:off+Dims], centroids)
		assignments[i] = int32(c)
		counts[c]++
	}

	// --- Build subCluster descriptors with absolute start offsets ---
	clusters := make([]subCluster, k)
	var cum int32 = int32(pStart)
	for c := 0; c < k; c++ {
		quantizeCentroid(&clusters[c].centroid, centroids[c])
		clusters[c].start = cum
		clusters[c].size = counts[c]
		cum += counts[c]
	}

	// --- Reorder data/labels within the partition by cluster assignment ---
	reorderPartitionByCluster(data, labels, pStart, n, assignments, counts, k)

	return clusters
}

// nearestCentroidIdx returns the index of the nearest float32 centroid for vec.
// Uses per-centroid early exit once current distance exceeds best.
func nearestCentroidIdx(vec []uint8, centroids [][Dims]float32) int {
	best := 0
	bestDist := float32(1e38)
	for c := range centroids {
		var d float32
		for i := 0; i < Dims; i++ {
			diff := float32(vec[i]) - centroids[c][i]
			d += diff * diff
			if d >= bestDist {
				goto nextCentroid
			}
		}
		if d < bestDist {
			bestDist = d
			best = c
		}
	nextCentroid:
	}
	return best
}

func quantizeCentroid(dst *[Dims]uint8, src [Dims]float32) {
	for d := 0; d < Dims; d++ {
		v := src[d]
		if v <= 0 {
			dst[d] = 0
		} else if v >= 255 {
			dst[d] = 255
		} else {
			dst[d] = uint8(v)
		}
	}
}

func centroidOfRange(data []uint8, pStart, pEnd int) [Dims]uint8 {
	var c [Dims]uint8
	if pStart < pEnd {
		off := pStart * Dims
		copy(c[:], data[off:off+Dims])
	}
	return c
}

// reorderPartitionByCluster reorders data and labels within the partition [pStart, pStart+n)
// so that all vectors belonging to the same cluster are contiguous.
// assignments[i] is the cluster index for vector at position pStart+i.
// counts[c] is the number of vectors in cluster c.
// After reordering, cluster c occupies positions [sum(counts[:c]), sum(counts[:c+1])) relative to pStart.
func reorderPartitionByCluster(data []uint8, labels []bool, pStart, n int, assignments []int32, counts []int32, k int) {
	// Compute output start offsets per cluster (relative to partition start).
	starts := make([]int32, k)
	var cum int32
	for c := 0; c < k; c++ {
		starts[c] = cum
		cum += counts[c]
	}

	newData := make([]uint8, n*Dims)
	newLabels := make([]bool, n)
	cursors := make([]int32, k)
	copy(cursors, starts)

	for i := 0; i < n; i++ {
		c := assignments[i]
		pos := int(cursors[c])
		srcOff := (pStart + i) * Dims
		copy(newData[pos*Dims:(pos+1)*Dims], data[srcOff:srcOff+Dims])
		newLabels[pos] = labels[pStart+i]
		cursors[c]++
	}

	copy(data[pStart*Dims:(pStart+n)*Dims], newData)
	copy(labels[pStart:pStart+n], newLabels)
}

// centroidDist computes integer squared distance from query to a quantized centroid.
// Dimensions where the query has SentinelU8 (missing) are skipped.
func centroidDist(q, c [Dims]uint8) uint32 {
	var sum uint32
	for d := 0; d < Dims; d++ {
		qd := q[d]
		if qd == SentinelU8 {
			continue
		}
		diff := int32(qd) - int32(c[d])
		sum += uint32(diff * diff)
	}
	return sum
}

// findTopClusters returns the indices of the kSearchClusters nearest clusters.
// Uses a linear scan over all clusters (at most kSubClusters=64, fits in L1 cache).
func findTopClusters(q [Dims]uint8, clusters []subCluster) [kSearchClusters]int {
	var topIdx [kSearchClusters]int
	var topDist [kSearchClusters]uint32

	n := kSearchClusters
	if n > len(clusters) {
		n = len(clusters)
	}

	filled := 0
	worstIdx := 0
	worstDist := uint32(0)

	for i := range clusters {
		d := centroidDist(q, clusters[i].centroid)
		if filled < n {
			topIdx[filled] = i
			topDist[filled] = d
			filled++
			if filled == n {
				worstIdx, worstDist = findWorst(topDist[:n])
			}
		} else if d < worstDist {
			topIdx[worstIdx] = i
			topDist[worstIdx] = d
			worstIdx, worstDist = findWorst(topDist[:n])
		}
	}

	for i := filled; i < kSearchClusters; i++ {
		topIdx[i] = -1
	}

	for i := 1; i < filled; i++ {
		for j := i; j > 0 && topDist[j] < topDist[j-1]; j-- {
			topDist[j], topDist[j-1] = topDist[j-1], topDist[j]
			topIdx[j], topIdx[j-1] = topIdx[j-1], topIdx[j]
		}
	}

	return topIdx
}

func findWorst(a []uint32) (int, uint32) {
	idx := 0
	worst := a[0]
	for i := 1; i < len(a); i++ {
		if a[i] > worst {
			worst = a[i]
			idx = i
		}
	}
	return idx, worst
}
