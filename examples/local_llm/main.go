// Command local_llm runs retrieval-augmented generation entirely on local
// servers. raggo parses and chunks the documents, embeds them through an
// OpenAI-compatible /v1/embeddings endpoint (llama.cpp or KoboldCpp), stores
// and retrieves them with Redis hybrid search, and the local chat model answers
// from the retrieved chunks.
//
//	REDIS_ADDR=localhost:6379 \
//	EMBED_URL=http://localhost:8081/v1/embeddings \
//	CHAT_URL=http://localhost:8080/v1/chat/completions \
//	go run ./examples/local_llm -docs examples/chat/docs -q "What is a vector database?"
//
// For a hosted OpenAI-compatible API such as Gemini, also set API_KEY,
// EMBED_MODEL and CHAT_MODEL (local servers ignore all three).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/teilomillet/raggo"
)

const collection = "local_docs"

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	docs := flag.String("docs", "examples/chat/docs", "directory of .txt and .pdf files to index")
	question := flag.String("q", "What did the PressureValve system do during Black Friday?", "question to answer")
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
	defer cancel()
	if err := run(ctx, *docs, *question, config{
		redisAddr:  env("REDIS_ADDR", "localhost:6379"),
		embedURL:   env("EMBED_URL", "http://localhost:8081/v1/embeddings"),
		chatURL:    env("CHAT_URL", "http://localhost:8080/v1/chat/completions"),
		apiKey:     env("API_KEY", "none"), // raggo requires a non-empty key
		embedModel: env("EMBED_MODEL", "local"),
		chatModel:  env("CHAT_MODEL", ""), // omitted from the request when empty
	}); err != nil {
		log.Fatal(err)
	}
}

// config holds the endpoints and credentials; see main for the env variables.
type config struct {
	redisAddr, embedURL, chatURL, apiKey, embedModel, chatModel string
}

func run(ctx context.Context, docsDir, question string, cfg config) error {
	// "openai" means any OpenAI-compatible server. Local servers ignore the model
	// name and key, but raggo requires a non-empty key.
	embedder, err := raggo.NewEmbedder(
		raggo.SetEmbedderProvider("openai"),
		raggo.SetEmbedderModel(cfg.embedModel),
		raggo.SetEmbedderAPIKey(cfg.apiKey),
		raggo.SetOption("api_url", cfg.embedURL),
	)
	if err != nil {
		return err
	}
	// Size the index from the model instead of assuming 1536.
	probe, err := embedder.Embed(ctx, "dimension probe")
	if err != nil {
		return fmt.Errorf("embeddings endpoint %s: %w", cfg.embedURL, err)
	}

	db, err := raggo.NewVectorDB(raggo.WithType("redis"), raggo.WithAddress(cfg.redisAddr))
	if err != nil {
		return err
	}
	if err := db.Connect(ctx); err != nil {
		return err
	}
	defer db.Close()

	if err := index(ctx, db, embedder, docsDir, len(probe)); err != nil {
		return err
	}

	db.SetColumnNames([]string{"Text", "Metadata"})
	qvec, err := embedder.Embed(ctx, question)
	if err != nil {
		return err
	}
	results, err := db.HybridSearch(ctx, collection, map[string]raggo.Vector{"Embedding": qvec}, 3, "COSINE",
		map[string]interface{}{"query_text": question, "ef": 64}, nil)
	if err != nil {
		return err
	}
	fmt.Println("Sources:")
	for _, r := range results {
		fmt.Printf("  %.3f %v\n", r.Score, r.Fields["Metadata"])
	}

	answer, err := chat(ctx, cfg, question, results)
	if err != nil {
		return err
	}
	fmt.Println("Answer:", answer)
	return nil
}

// index parses, chunks and embeds every .txt and .pdf file in dir, and only
// then replaces the collection with one sized to the embedding model. Nothing
// is dropped if there is nothing to index or an embedding fails.
func index(ctx context.Context, db *raggo.VectorDB, embedder raggo.Embedder, dir string, dim int) error {
	parser := raggo.NewParser()
	chunker, err := raggo.NewChunker(raggo.ChunkSize(200), raggo.ChunkOverlap(50))
	if err != nil {
		return err
	}
	files, err := filepath.Glob(filepath.Join(dir, "*"))
	if err != nil {
		return err
	}
	var records []raggo.Record
	for _, f := range files {
		if ext := strings.ToLower(filepath.Ext(f)); ext != ".txt" && ext != ".pdf" {
			continue
		}
		doc, err := parser.Parse(f)
		if err != nil {
			return fmt.Errorf("parse %s: %w", f, err)
		}
		for _, c := range chunker.Chunk(doc.Content) {
			vec, err := embedder.Embed(ctx, c.Text)
			if err != nil {
				return fmt.Errorf("embed %s: %w", f, err)
			}
			records = append(records, raggo.Record{Fields: map[string]interface{}{
				"Embedding": vec, "Text": c.Text, "Metadata": map[string]interface{}{"source": filepath.Base(f)},
			}})
		}
	}
	if len(records) == 0 {
		return fmt.Errorf("no .txt or .pdf files in %s", dir)
	}

	if ok, err := db.HasCollection(ctx, collection); err != nil {
		return err
	} else if ok {
		if err := db.DropCollection(ctx, collection); err != nil {
			return err
		}
	}
	if err := db.CreateCollection(ctx, collection, raggo.Schema{Name: collection, Fields: []raggo.Field{
		{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
		{Name: "Embedding", DataType: "float_vector", Dimension: dim},
		{Name: "Text", DataType: "varchar", MaxLength: 65535},
		{Name: "Metadata", DataType: "varchar", MaxLength: 65535},
	}}); err != nil {
		return err
	}
	if err := db.CreateIndex(ctx, collection, "Embedding", raggo.Index{Type: "HNSW", Metric: "COSINE",
		Parameters: map[string]interface{}{"M": 16, "efConstruction": 200}}); err != nil {
		return err
	}
	if err := db.Insert(ctx, collection, records); err != nil {
		return err
	}
	fmt.Printf("Indexed %d chunks (%d-dim embeddings)\n", len(records), dim)
	return nil
}

// chat asks the local model to answer from the retrieved chunks only.
func chat(ctx context.Context, cfg config, question string, results []raggo.SearchResult) (string, error) {
	url := cfg.chatURL
	var sb strings.Builder
	for i, r := range results {
		fmt.Fprintf(&sb, "[%d] %v\n\n", i+1, r.Fields["Text"])
	}
	payload := map[string]interface{}{
		"messages": []map[string]string{
			{"role": "system", "content": "Answer using only the provided context. If it does not contain the answer, say you don't know."},
			{"role": "user", "content": "Context:\n" + sb.String() + "Question: " + question},
		},
		"temperature": 0.2,
	}
	if cfg.chatModel != "" {
		payload["model"] = cfg.chatModel
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.apiKey) // same header raggo's embedder sends
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("chat endpoint %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return "", fmt.Errorf("chat endpoint %s: HTTP %d: %s", url, resp.StatusCode, msg)
	}
	var out struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", err
	}
	if len(out.Choices) == 0 {
		return "", fmt.Errorf("chat endpoint %s: no choices in reply", url)
	}
	return out.Choices[0].Message.Content, nil
}
