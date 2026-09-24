// Command redis_quickstart indexes a folder into Redis and runs a hybrid
// (BM25 + vector) query with raggo.RAG, using embeddings from a local
// OpenAI-compatible server such as llama.cpp or KoboldCpp.
//
//	docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4
//	llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --device none --port 8081
//	go run ./examples/redis_quickstart
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/teilomillet/raggo"
)

func main() {
	embedURL := os.Getenv("EMBED_URL")
	if embedURL == "" {
		embedURL = "http://localhost:8081/v1/embeddings"
	}
	ctx := context.Background()

	r, err := raggo.NewRAG(
		raggo.WithRedis("quickstart"), // Redis 8.4+ at localhost:6379; hybrid search is on by default
		raggo.SetEmbedURL(embedURL),   // any OpenAI-compatible /v1/embeddings endpoint
		raggo.SetAPIKey("none"),       // local servers ignore it, but raggo requires one
	)
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()

	// The index is sized from the model (768 for embeddinggemma-300M).
	// Re-running adds the documents again; use a new collection name to start fresh.
	if err := r.LoadDocuments(ctx, "examples/chat/docs"); err != nil {
		log.Fatal(err)
	}

	results, err := r.Query(ctx, "What did the PressureValve system do during Black Friday?")
	if err != nil {
		log.Fatal(err)
	}
	for _, res := range results {
		first, _, _ := strings.Cut(strings.TrimSpace(res.Content), "\n")
		fmt.Printf("%.3f  %s\n", res.Score, first)
	}
}
