//go:build integration

// End-to-end RAG on Redis with Gemini. raggo does the loading, chunking,
// embedding, storage and retrieval (Redis FT.SEARCH / FT.HYBRID). Gemini is
// reached through a test-registered "gemini" embedder, since raggo ships only
// an OpenAI one. The answer step calls Gemini directly because gollm v0.1.1
// has no Gemini provider.
//
//	docker run -d --rm -p 6379:6379 redis:8.4
//	REDIS_ADDR=localhost:6379 GEMINI_API_KEY=... go test -tags=integration -run TestGeminiRAG -v .
package raggo_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"
	"unicode"

	"github.com/redis/go-redis/v9"
	"github.com/teilomillet/raggo"
	"github.com/teilomillet/raggo/rag/providers"
)

const (
	geminiBase       = "https://generativelanguage.googleapis.com/v1beta/openai"
	geminiEmbedModel = "gemini-embedding-001"
	geminiChatModel  = "gemini-3.6-flash"
	geminiDims       = 1536 // raggo's RAG schema hardcodes 1536; Gemini truncates to it
	itCollection     = "raggo_it_gemini"
)

// geminiEmbedder implements providers.Embedder against Gemini's OpenAI-compatible API.
// It asks for 1536 dimensions, which raggo's built-in OpenAI embedder can't do.
type geminiEmbedder struct{ apiKey, model string }

func (g geminiEmbedder) Embed(ctx context.Context, text string) ([]float64, error) {
	var resp struct {
		Data []struct {
			Embedding []float64 `json:"embedding"`
		} `json:"data"`
	}
	err := geminiPost(ctx, g.apiKey, "/embeddings", map[string]interface{}{
		"model": g.model, "input": text, "dimensions": geminiDims,
	}, &resp)
	if err != nil {
		return nil, err
	}
	if len(resp.Data) != 1 {
		return nil, fmt.Errorf("gemini embeddings: got %d vectors, want 1", len(resp.Data))
	}
	return resp.Data[0].Embedding, nil
}

func (g geminiEmbedder) GetDimension() (int, error) { return geminiDims, nil }

// geminiPost POSTs JSON, retrying on 429/5xx so free-tier rate limits don't flake the test.
func geminiPost(ctx context.Context, apiKey, path string, body, out interface{}) error {
	payload, err := json.Marshal(body)
	if err != nil {
		return err
	}
	var lastErr error
	for attempt := 1; attempt <= 4; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, geminiBase+path, bytes.NewReader(payload))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Authorization", "Bearer "+apiKey)
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			return err
		}
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return err
		}
		if resp.StatusCode == http.StatusOK {
			return json.Unmarshal(data, out)
		}
		lastErr = fmt.Errorf("gemini %s: HTTP %d: %.300s", path, resp.StatusCode, data)
		if resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode < 500 {
			return lastErr
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt) * 3 * time.Second):
		}
	}
	return lastErr
}

// geminiAnswer is the generation step: answer from the retrieved context only.
func geminiAnswer(ctx context.Context, apiKey, question string, results []raggo.RetrieverResult) (string, error) {
	var sb strings.Builder
	for i, r := range results {
		fmt.Fprintf(&sb, "[%d] %s\n\n", i+1, r.Content)
	}
	var resp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	err := geminiPost(ctx, apiKey, "/chat/completions", map[string]interface{}{
		"model": geminiChatModel,
		"messages": []map[string]string{
			{"role": "system", "content": "Answer using only the provided context. If the context does not contain the answer, say you don't know."},
			{"role": "user", "content": "Context:\n" + sb.String() + "\nQuestion: " + question},
		},
	}, &resp)
	if err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("gemini chat: no choices")
	}
	return resp.Choices[0].Message.Content, nil
}

func dropCollection(t *testing.T, addr, col string) {
	t.Helper()
	ctx := context.Background()
	db, err := raggo.NewVectorDB(raggo.WithType("redis"), raggo.WithAddress(addr))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	if err := db.Connect(ctx); err != nil {
		t.Fatal(err)
	}
	if ok, err := db.HasCollection(ctx, col); err != nil {
		t.Fatal(err)
	} else if ok {
		if err := db.DropCollection(ctx, col); err != nil {
			t.Fatal(err)
		}
	}
}

func contents(results []raggo.RetrieverResult) string {
	var sb strings.Builder
	for _, r := range results {
		sb.WriteString(strings.ToLower(r.Content))
		sb.WriteString("\n")
	}
	return sb.String()
}

func TestGeminiRAGEndToEnd(t *testing.T) {
	addr, key := os.Getenv("REDIS_ADDR"), os.Getenv("GEMINI_API_KEY")
	if addr == "" || key == "" {
		t.Skip("set REDIS_ADDR (Redis 8.4+) and GEMINI_API_KEY to run")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	providers.RegisterEmbedder("gemini", func(cfg map[string]interface{}) (providers.Embedder, error) {
		apiKey, _ := cfg["api_key"].(string)
		model, _ := cfg["model"].(string)
		if apiKey == "" || model == "" {
			return nil, fmt.Errorf("gemini embedder needs api_key and model, got %v", cfg)
		}
		return geminiEmbedder{apiKey: apiKey, model: model}, nil
	})

	dropCollection(t, addr, itCollection)
	t.Cleanup(func() { dropCollection(t, addr, itCollection) })

	newRAG := func(t *testing.T, strategy string) *raggo.RAG {
		t.Helper()
		r, err := raggo.NewRAG(
			raggo.SetProvider("gemini"),
			raggo.SetModel(geminiEmbedModel),
			raggo.SetAPIKey(key),
			raggo.SetDBType("redis"),
			raggo.SetDBAddress(addr),
			raggo.SetCollection(itCollection),
			raggo.SetSearchStrategy(strategy),
			raggo.SetTopK(3),
			// MinScore stays at the library default (0.7) on purpose: Review Focus 4.
		)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { r.Close() })
		return r
	}

	hybrid := newRAG(t, "hybrid")
	if err := hybrid.LoadDocuments(ctx, "examples/chat/docs"); err != nil {
		t.Fatalf("LoadDocuments: %v", err)
	}
	dense := newRAG(t, "dense") // same collection, vector-only search

	retriever, err := raggo.NewRetriever(
		raggo.WithRetrieveDB("redis", addr),
		raggo.WithRetrieveCollection(itCollection),
		raggo.WithRetrieveEmbedding("gemini", geminiEmbedModel, key),
		raggo.WithTopK(3),
		raggo.WithHybrid(true),
	)
	if err != nil {
		t.Fatalf("NewRetriever: %v", err)
	}
	t.Cleanup(func() { retriever.Close() })

	queries := []struct {
		name  string
		query string
		want  string // lowercase substring expected in the retrieved text
	}{
		{name: "vector databases", query: "How do vector databases perform similarity search?", want: "vector database"},
		{name: "go concurrency", query: "How does Go handle concurrency with goroutines?", want: "goroutine"},
		{name: "microservices", query: "How do microservices communicate with each other?", want: "microservice"},
		{name: "fictional fact", query: "What did the PressureValve system do during Black Friday?", want: "pressurevalve"},
	}

	for _, tc := range queries {
		t.Run("hybrid/"+tc.name, func(t *testing.T) {
			res, err := hybrid.Query(ctx, tc.query)
			if err != nil {
				t.Fatal(err)
			}
			if len(res) == 0 {
				t.Fatal("no results: hybrid scores fell below the default MinScore 0.7")
			}
			if !strings.Contains(strings.ToLower(res[0].Content), tc.want) {
				t.Errorf("top result lacks %q (score %.3f): %.200q", tc.want, res[0].Score, res[0].Content)
			}
			for _, r := range res {
				if r.Score < 0 || r.Score > 1+1e-9 {
					t.Errorf("score %v outside [0,1]", r.Score)
				}
			}
		})
		t.Run("dense/"+tc.name, func(t *testing.T) {
			res, err := dense.Query(ctx, tc.query)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(contents(res), tc.want) {
				t.Errorf("dense top-3 lacks %q; got %d results", tc.want, len(res))
			}
		})
		t.Run("retriever/"+tc.name, func(t *testing.T) {
			res, err := retriever.Retrieve(ctx, tc.query)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(contents(res), tc.want) {
				t.Errorf("retriever top-3 lacks %q; got %d results", tc.want, len(res))
			}
		})
	}

	t.Run("generation grounded in retrieved context", func(t *testing.T) {
		const q = "What did MountainPass's PressureValve system do during Black Friday?"
		res, err := hybrid.Query(ctx, q)
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 0 {
			t.Fatal("no context retrieved")
		}
		answer, err := geminiAnswer(ctx, key, q, res)
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answer: %s", answer)
		// PressureValve is fictional: the model can only know it scales/balances load via retrieval.
		lower := strings.ToLower(answer)
		if !strings.Contains(lower, "load") && !strings.Contains(lower, "scal") && !strings.Contains(lower, "traffic") {
			t.Errorf("answer does not reflect the retrieved document: %q", answer)
		}
	})
}

const pvQuestion = "What did the PressureValve system do during Black Friday?"

func redisAddrOrSkip(t *testing.T) string {
	t.Helper()
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		t.Skip("set REDIS_ADDR (Redis 8.4+) to run")
	}
	return addr
}

// bowVector is a deterministic bag-of-words embedding: each lowercased word is
// hashed into one of dim buckets, then the vector is L2-normalized. Texts that
// share words score high, which makes retrieval assertions meaningful offline.
func bowVector(text string, dim int) []float64 {
	v := make([]float64, dim)
	words := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	for _, w := range words {
		h := fnv.New32a()
		h.Write([]byte(w))
		v[h.Sum32()%uint32(dim)]++
	}
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	if norm > 0 {
		norm = math.Sqrt(norm)
		for i := range v {
			v[i] /= norm
		}
	}
	return v
}

// bowEmbedder serves bowVector through raggo's provider registry.
type bowEmbedder struct{ dim int }

func (b bowEmbedder) Embed(_ context.Context, text string) ([]float64, error) {
	return bowVector(text, b.dim), nil
}

func (b bowEmbedder) GetDimension() (int, error) { return b.dim, nil }

// indexDim reads the vector DIM of a collection's FT index.
func indexDim(t *testing.T, addr, col string) int64 {
	t.Helper()
	c := redis.NewClient(&redis.Options{Addr: addr, Protocol: 2})
	defer c.Close()
	reply, err := c.Do(context.Background(), "FT.INFO", col).Slice()
	if err != nil {
		t.Fatalf("FT.INFO %s: %v", col, err)
	}
	for i := 0; i+1 < len(reply); i += 2 {
		if reply[i] != "attributes" {
			continue
		}
		attrs, _ := reply[i+1].([]interface{})
		for _, a := range attrs {
			kv, _ := a.([]interface{})
			for j := 0; j+1 < len(kv); j += 2 {
				if kv[j] == "dim" {
					if d, ok := kv[j+1].(int64); ok {
						return d
					}
				}
			}
		}
	}
	t.Fatalf("FT.INFO %s: no vector dim found", col)
	return 0
}

func newFake1536RAG(t *testing.T, addr, col string) *raggo.RAG {
	t.Helper()
	providers.RegisterEmbedder("fake1536", func(map[string]interface{}) (providers.Embedder, error) {
		return bowEmbedder{dim: 1536}, nil
	})
	dropCollection(t, addr, col)
	t.Cleanup(func() { dropCollection(t, addr, col) })
	r, err := raggo.NewRAG(
		raggo.SetProvider("fake1536"),
		raggo.SetAPIKey("none"),
		raggo.SetDBType("redis"),
		raggo.SetDBAddress(addr),
		raggo.SetCollection(col),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { r.Close() })
	return r
}

// Characterization: LoadDocuments creates a 1536-dim index and hybrid Query
// returns the PressureValve chunk first.
func TestRAGCharacterizeLoadAndQuery(t *testing.T) {
	addr := redisAddrOrSkip(t)
	ctx := t.Context() // cancelled when the test ends (Go 1.24+)
	const col = "raggo_it_char_load"
	r := newFake1536RAG(t, addr, col)

	if err := r.LoadDocuments(ctx, "examples/chat/docs"); err != nil {
		t.Fatalf("LoadDocuments: %v", err)
	}
	if d := indexDim(t, addr, col); d != 1536 {
		t.Errorf("index dim = %d, want 1536", d)
	}
	res, err := r.Query(ctx, pvQuestion)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || !strings.Contains(strings.ToLower(res[0].Content), "pressurevalve") {
		t.Errorf("top result is not the PressureValve chunk: %+v", res)
	}
}

// Characterization: ProcessWithContext creates a 1536-dim index before it
// parses the source, so a missing file fails after the index exists.
func TestRAGCharacterizeProcessWithContextSchema(t *testing.T) {
	addr := redisAddrOrSkip(t)
	const col = "raggo_it_char_pwc"
	r := newFake1536RAG(t, addr, col)

	err := r.ProcessWithContext(t.Context(), "does-not-exist.txt", "gpt-4o-mini")
	if err == nil || !strings.Contains(err.Error(), "failed to parse document") {
		t.Fatalf("err = %v, want a parse error after collection creation", err)
	}
	if d := indexDim(t, addr, col); d != 1536 {
		t.Errorf("index dim = %d, want 1536", d)
	}
}

// raggo.RAG with a local-style 768-dim OpenAI-compatible endpoint on Redis:
// the index is sized from the model and hybrid search finds the right chunk.
func TestRAGLocalEmbeddingsOnRedis(t *testing.T) {
	addr := redisAddrOrSkip(t)
	ctx := t.Context()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		var req struct {
			Input string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{{"embedding": bowVector(req.Input, 768)}},
		})
	}))
	defer srv.Close()

	const col = "raggo_it_local768"
	dropCollection(t, addr, col)
	t.Cleanup(func() { dropCollection(t, addr, col) })
	r, err := raggo.NewRAG(
		raggo.SetDBType("redis"),
		raggo.SetDBAddress(addr),
		raggo.SetCollection(col),
		raggo.SetEmbedURL(srv.URL+"/v1/embeddings"),
		raggo.SetAPIKey("none"),
	)
	if err != nil {
		t.Fatal(err)
	}
	defer r.Close()

	if err := r.LoadDocuments(ctx, "examples/chat/docs"); err != nil {
		t.Fatalf("LoadDocuments: %v", err)
	}
	if d := indexDim(t, addr, col); d != 768 {
		t.Errorf("index dim = %d, want 768", d)
	}
	res, err := r.Query(ctx, pvQuestion)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) == 0 || !strings.Contains(strings.ToLower(res[0].Content), "pressurevalve") {
		t.Errorf("top result is not the PressureValve chunk: %+v", res)
	}
	if calls.Load() == 0 {
		t.Error("embeddings server was never called: EmbedURL not used")
	}
}
