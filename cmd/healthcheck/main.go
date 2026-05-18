package main

import (
	"net/http"
	"os"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}
	resp, err := http.Get("http://localhost:" + port + "/ready")
	if err != nil || resp.StatusCode != 200 {
		os.Exit(1)
	}
}
