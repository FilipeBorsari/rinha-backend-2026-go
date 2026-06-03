package vectorstore

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"sort"
)

var snapshotMagic = [8]byte{'R', 'V', 'S', 'N', 'A', 'P', '0', '1'}

const snapshotVersion uint32 = 1

func WriteSnapshot(path string, store *VectorStore) (retErr error) {
	tmpPath := path + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return err
	}
	defer func() {
		if retErr != nil {
			_ = os.Remove(tmpPath)
		}
	}()

	bw := bufio.NewWriterSize(f, 1<<20)
	if err := writeSnapshot(bw, store); err != nil {
		_ = f.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		_ = f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return err
	}
	return nil
}

func loadSnapshot(path string) (*VectorStore, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	br := bufio.NewReaderSize(f, 1<<20)
	return readSnapshot(br)
}

func writeSnapshot(w io.Writer, store *VectorStore) error {
	if store == nil {
		return fmt.Errorf("store is nil")
	}
	if store.n != len(store.labels) {
		return fmt.Errorf("snapshot labels mismatch: n=%d labels=%d", store.n, len(store.labels))
	}
	if len(store.data) != store.n*Dims {
		return fmt.Errorf("snapshot data mismatch: len=%d expected=%d", len(store.data), store.n*Dims)
	}

	if _, err := w.Write(snapshotMagic[:]); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, snapshotVersion); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(Dims)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(len(store.l1s))); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint32(L2PerL1)); err != nil {
		return err
	}
	if err := binary.Write(w, binary.LittleEndian, uint64(store.n)); err != nil {
		return err
	}

	if _, err := w.Write(store.data); err != nil {
		return err
	}

	labelBytes := make([]byte, len(store.labels))
	for i, fraud := range store.labels {
		if fraud {
			labelBytes[i] = 1
		}
	}
	if _, err := w.Write(labelBytes); err != nil {
		return err
	}

	normValues := [7]float32{
		store.normConsts.MaxAmount,
		store.normConsts.MaxInstallments,
		store.normConsts.AmountVsAvgRatio,
		store.normConsts.MaxMinutes,
		store.normConsts.MaxKm,
		store.normConsts.MaxTxCount24h,
		store.normConsts.MaxMerchantAvg,
	}
	if err := binary.Write(w, binary.LittleEndian, normValues); err != nil {
		return err
	}

	keys := make([]string, 0, len(store.mccRisk))
	for key := range store.mccRisk {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	if err := binary.Write(w, binary.LittleEndian, uint32(len(keys))); err != nil {
		return err
	}
	for _, key := range keys {
		if err := writeString(w, key); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, store.mccRisk[key]); err != nil {
			return err
		}
	}

	for _, l1 := range store.l1s {
		if _, err := w.Write(l1.centroid[:]); err != nil {
			return err
		}
		for _, l2 := range l1.l2s {
			if _, err := w.Write(l2.centroid[:]); err != nil {
				return err
			}
			if err := binary.Write(w, binary.LittleEndian, l2.start); err != nil {
				return err
			}
			if err := binary.Write(w, binary.LittleEndian, l2.length); err != nil {
				return err
			}
		}
	}

	return nil
}

func readSnapshot(r io.Reader) (*VectorStore, error) {
	var magic [8]byte
	if _, err := io.ReadFull(r, magic[:]); err != nil {
		return nil, err
	}
	if magic != snapshotMagic {
		return nil, fmt.Errorf("%w: bad magic", ErrSnapshotInvalid)
	}

	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if version != snapshotVersion {
		return nil, fmt.Errorf("%w: unsupported version %d", ErrSnapshotInvalid, version)
	}

	var dims uint32
	if err := binary.Read(r, binary.LittleEndian, &dims); err != nil {
		return nil, err
	}
	if dims != Dims {
		return nil, fmt.Errorf("%w: unsupported dims %d", ErrSnapshotInvalid, dims)
	}

	var l1Count uint32
	if err := binary.Read(r, binary.LittleEndian, &l1Count); err != nil {
		return nil, err
	}

	var l2PerL1 uint32
	if err := binary.Read(r, binary.LittleEndian, &l2PerL1); err != nil {
		return nil, err
	}
	if l2PerL1 != L2PerL1 {
		return nil, fmt.Errorf("%w: unsupported l2-per-l1 %d", ErrSnapshotInvalid, l2PerL1)
	}

	var count uint64
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}
	if count > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("%w: snapshot too large", ErrSnapshotInvalid)
	}

	n := int(count)
	data := make([]uint8, n*Dims)
	if _, err := io.ReadFull(r, data); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%w: truncated data", ErrSnapshotInvalid)
		}
		return nil, err
	}

	labelBytes := make([]byte, n)
	if _, err := io.ReadFull(r, labelBytes); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%w: truncated labels", ErrSnapshotInvalid)
		}
		return nil, err
	}
	labels := make([]bool, n)
	for i, b := range labelBytes {
		switch b {
		case 0:
		case 1:
			labels[i] = true
		default:
			return nil, fmt.Errorf("%w: invalid label byte %d", ErrSnapshotInvalid, b)
		}
	}

	var normValues [7]float32
	if err := binary.Read(r, binary.LittleEndian, &normValues); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%w: truncated normalization constants", ErrSnapshotInvalid)
		}
		return nil, err
	}
	norm := NormConstants{
		MaxAmount:        normValues[0],
		MaxInstallments:  normValues[1],
		AmountVsAvgRatio: normValues[2],
		MaxMinutes:       normValues[3],
		MaxKm:            normValues[4],
		MaxTxCount24h:    normValues[5],
		MaxMerchantAvg:   normValues[6],
	}

	var mccCount uint32
	if err := binary.Read(r, binary.LittleEndian, &mccCount); err != nil {
		if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
			return nil, fmt.Errorf("%w: truncated MCC header", ErrSnapshotInvalid)
		}
		return nil, err
	}
	mccRisk := make(map[string]float32, mccCount)
	for i := uint32(0); i < mccCount; i++ {
		key, err := readString(r)
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, fmt.Errorf("%w: truncated MCC payload", ErrSnapshotInvalid)
			}
			return nil, err
		}

		var value float32
		if err := binary.Read(r, binary.LittleEndian, &value); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, fmt.Errorf("%w: truncated MCC value", ErrSnapshotInvalid)
			}
			return nil, err
		}
		mccRisk[key] = value
	}

	l1s := make([]L1Cluster, l1Count)
	for i := range l1s {
		if _, err := io.ReadFull(r, l1s[i].centroid[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil, fmt.Errorf("%w: truncated L1 centroid", ErrSnapshotInvalid)
			}
			return nil, err
		}
		for j := range l1s[i].l2s {
			if _, err := io.ReadFull(r, l1s[i].l2s[j].centroid[:]); err != nil {
				if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
					return nil, fmt.Errorf("%w: truncated L2 centroid", ErrSnapshotInvalid)
				}
				return nil, err
			}
			if err := binary.Read(r, binary.LittleEndian, &l1s[i].l2s[j].start); err != nil {
				if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
					return nil, fmt.Errorf("%w: truncated L2 start", ErrSnapshotInvalid)
				}
				return nil, err
			}
			if err := binary.Read(r, binary.LittleEndian, &l1s[i].l2s[j].length); err != nil {
				if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
					return nil, fmt.Errorf("%w: truncated L2 length", ErrSnapshotInvalid)
				}
				return nil, err
			}
		}
	}

	return &VectorStore{
		data:       data,
		labels:     labels,
		n:          n,
		mccRisk:    mccRisk,
		normConsts: norm,
		l1s:        l1s,
	}, nil
}

func writeString(w io.Writer, value string) error {
	if len(value) > int(^uint16(0)) {
		return fmt.Errorf("string too long: %d", len(value))
	}
	if err := binary.Write(w, binary.LittleEndian, uint16(len(value))); err != nil {
		return err
	}
	_, err := io.WriteString(w, value)
	return err
}

func readString(r io.Reader) (string, error) {
	var size uint16
	if err := binary.Read(r, binary.LittleEndian, &size); err != nil {
		return "", err
	}
	buf := make([]byte, size)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}
