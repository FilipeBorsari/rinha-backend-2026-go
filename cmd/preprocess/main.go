package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/filipeborsari/rinha-de-backend-2026-go/internal/vectorstore"
)

func main() {
	refsPath := flag.String("refs", "resources/references.json.gz", "path to the compressed references dataset")
	mccPath := flag.String("mcc", "resources/mcc_risk.json", "path to the MCC risk file")
	normPath := flag.String("norm", "resources/normalization.json", "path to the normalization constants file")
	outPath := flag.String("out", "resources/references.bin", "path to the generated snapshot")
	flag.Parse()

	store, err := vectorstore.Build(*refsPath, *mccPath, *normPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if err := os.MkdirAll(filepath.Dir(*outPath), 0o755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	if err := vectorstore.WriteSnapshot(*outPath, store); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
