# Redis Minor Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Execution method chosen:** Native (superpowers:executing-plans), with one whole-branch review at the end.
> After approval, copy this file to `docs/superpowers/plans/2026-09-24-redis-minor-fixes.md` and commit it with Task 1.

**Goal:** Fix and test the minor items deferred from the PR #2 review. Then add a USAGE.md, backed by a runnable example, showing RAG with raggo on local llama.cpp or KoboldCpp servers with Redis. Finish with the required Gemini end-to-end test.

**Architecture:**
- **`efRuntime`** reuses `floatParam`, so any Go number works, and rejects values that aren't whole numbers from 1 to MaxInt32. `HybridSearch` reads it before its first Redis call, and `hybridArgs` takes the resolved `ef int`.
- **Weights under RRF:** checking `alpha`/`beta` even when they're unused becomes documented and tested.
- **Test layout:** `TestRedisOptions` moves to `rag_test.go`, the file named after the source it tests.

**Tech Stack:** Go 1.27.1, `github.com/redis/go-redis/v9`, Redis 8.4 (docker).

**Spec:** the "Left for later (minor)" list from PR #2:
1. `alpha`/`beta` are checked even when unused. The behaviour is kept but undocumented and untested.
2. `efRuntime` accepts only `int`, so a `float64` from JSON silently becomes 10.
3. The `int32` case of `floatParam` is untested. **Moot:** PR #2 replaced the type switch with `reflect`'s `CanInt`, which handles every signed int kind in one branch, and the `int8` row already exercises it. There's no separate `int32` code left to test.
4. Unit tests are not in the same order as the functions they test. **Won't fix, by the user's choice:** golang-testing rule 13 is a SHOULD, test results don't depend on order, and a reordering diff would add blame and review noise for readability alone.
5. `TestRedisOptions` sits in `contextual_rag_test.go` but tests `rag.go` options.

## Global Constraints
- **Start state:** local `main` (`43d8d0b`) is behind `origin/main` (`af50c10`, the PR #2 merge). Fast-forward it first, work on the new branch `redis-minor-fixes`, and never commit to `main`.
- **Build and vet:** use `go build . ./rag/... ./config/...` and `go vet [-tags=integration] . ./rag/... ./config/...`. Never use `./...`.
- **Commits:** every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus
1. **`ef` arriving as `float64(64)` from JSON config** must be used as 64, not 10. Pinned by `TestEfRuntime` (Task 1).
2. **A fractional, zero or huge `ef`** must be an error, never a silent 10 and never an `int` overflow. Pinned by `TestEfRuntime` (Task 1).
3. **A bad `ef` passed to `HybridSearch`** must fail before its first Redis call (`textMatches`). Pinned by the `ef` row of the never-connected table test (Task 2).
4. **A bad `alpha` under RRF, where the weights are unused,** must fail fast, and the doc comment must say so. Pinned by the RRF row (Task 2).
6. **A local embedding model that isn't 1536-dimensional** (embeddinggemma-300M is 768): the documented path must size the Redis index from the model, and the docs must steer users away from `raggo.RAG`'s fixed 1536. Pinned by the `768-dim` line in the llama.cpp and KoboldCpp runs (Task 4).
5. **The file move silently drops a check:** both `TestRedisOptions` (now in `rag_test.go`) and `TestDefaultContextualConfig` must appear as PASS in the verbose run (Task 3, Step 2).

---

### Task 1: `efRuntime` accepts any number and rejects bad values

**Files:**
- Modify: `rag/redis.go`:
  - `efRuntime` (~line 202)
  - `Search` (~line 473)
  - `HybridSearch` (~line 533)
  - `hybridArgs` (~line 624)
- Test: `rag/redis_test.go`. Put `TestEfRuntime` right before `TestFloatParam`.

**Interfaces:**
- Consumes: the existing `floatParam(params map[string]interface{}, key string, def float64) (float64, error)`.
- Produces:
  - `func efRuntime(params map[string]interface{}) (int, error)`
  - `func hybridArgs(index, query, field string, vec Vector, topK int, linear bool, alpha, beta float64, ef int, cols []string) []interface{}`

- [ ] **Step 0: Set up**

```bash
git checkout main && git pull --ff-only origin main   # 43d8d0b -> af50c10
git checkout -b redis-minor-fixes
mkdir -p docs/superpowers/plans && cp /home/azilber/.claude/plans/buzzing-brewing-valiant.md docs/superpowers/plans/2026-09-24-redis-minor-fixes.md
docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4
```

- [ ] **Step 1: Write the failing test.** Only `efRuntime`'s own rules are tested here; type handling is already covered by `TestFloatParam`.

```go
func TestEfRuntime(t *testing.T) {
	tests := []struct {
		name    string
		params  map[string]interface{}
		want    int
		wantErr bool
	}{
		{name: "absent uses Redis default", params: nil, want: 10},
		{name: "float64 from JSON config", params: map[string]interface{}{"ef": float64(64)}, want: 64},
		{name: "fractional is an error", params: map[string]interface{}{"ef": 64.5}, wantErr: true},
		{name: "zero is an error", params: map[string]interface{}{"ef": 0}, wantErr: true},
		{name: "above MaxInt32 is an error", params: map[string]interface{}{"ef": 1e12}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := efRuntime(tt.params)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), `"ef"`) {
					t.Errorf("efRuntime(%v) err = %v, want error naming \"ef\"", tt.params, err)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Errorf("efRuntime(%v) = %d, %v; want %d, nil", tt.params, got, err, tt.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `go test ./rag -run TestEfRuntime -v`
Expected: FAIL to compile with `assignment mismatch: 2 variables but efRuntime returns 1 value`.

- [ ] **Step 3: Implement**

Replace `efRuntime`:

```go
// efRuntime reads the HNSW search-time ef as any Go number (float64 when it
// comes from JSON config); absent means Redis's default, 10. Anything but a
// whole number in [1, MaxInt32] is an error, which also keeps int() safe.
func efRuntime(params map[string]interface{}) (int, error) {
	f, err := floatParam(params, "ef", 10)
	if err != nil {
		return 0, err
	}
	if f < 1 || f > math.MaxInt32 || f != math.Trunc(f) {
		return 0, fmt.Errorf("searchParams[%q] must be a whole number from 1 to %d, got %v", "ef", math.MaxInt32, params["ef"])
	}
	return int(f), nil
}
```

- **`Search`:** right after the `validName("field", field)` check, add `ef, err := efRuntime(searchParams)` followed by `if err != nil { return nil, err }`. Change the params literal to `"EF": ef`, and leave the later `res, err := ...` as it is (it's legal because `res` is new).
- **`HybridSearch`:** right after the `beta, err := floatParam(...)` block, add the same two lines. Change the `hybridArgs` call to pass `alpha, beta, ef, cols`.
- **`hybridArgs`:** change the signature to `(..., alpha, beta float64, ef int, cols []string)` and write `"EF_RUNTIME", ef`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `go vet ./rag && go vet -tags=integration ./rag && REDIS_ADDR=localhost:6379 go test -tags=integration -race -count=1 ./rag`
Expected: `ok`. The integration tests pass `"ef": 64` and `1000` as ints, which are still accepted.

- [ ] **Step 5: Commit**

```bash
git add rag/redis.go rag/redis_test.go docs/superpowers/plans/2026-09-24-redis-minor-fixes.md
git commit -m "fix(rag): efRuntime accepts any numeric ef and rejects bad values"
```

---

### Task 2: Pin fail-fast validation of unused weights and of `ef` in `HybridSearch`

**Files:**
- Modify: `rag/redis.go`, the `HybridSearch` doc comment (~lines 504-508).
- Test: `rag/redis_test.go`. Replace `TestRedisHybridSearchRejectsBadWeightBeforeRedis`.

- [ ] **Step 1: Turn the existing test into a table**

```go
// Bad params fail before any Redis call (this RedisDB was never connected, so
// reaching Redis would nil-deref), including weights RRF would never use.
func TestRedisHybridSearchRejectsBadParamsBeforeRedis(t *testing.T) {
	tests := []struct {
		name    string
		params  map[string]interface{}
		wantKey string
	}{
		{name: "string alpha with LINEAR", params: map[string]interface{}{"combine": "LINEAR", "alpha": "0.3"}, wantKey: "alpha"},
		{name: "string alpha with RRF (unused)", params: map[string]interface{}{"combine": "RRF", "alpha": "0.3"}, wantKey: "alpha"},
		{name: "fractional ef", params: map[string]interface{}{"ef": 64.5}, wantKey: `"ef"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			db, _ := newRedisDB(&Config{})
			_, err := db.HybridSearch(context.Background(), "docs", map[string]Vector{"Embedding": {1, 0}}, 3, "COSINE", tt.params, nil)
			if err == nil || !strings.Contains(err.Error(), tt.wantKey) {
				t.Errorf("err = %v, want error naming %s", err, tt.wantKey)
			}
		})
	}
}
```

- [ ] **Step 2: Prove the RRF row has teeth.** The behaviour already exists, so break the code, watch the row fail, then restore it:

```bash
go test ./rag -run TestRedisHybridSearchRejectsBadParamsBeforeRedis -v   # expect PASS
python3 - <<'EOF'
p='rag/redis.go'; s=open(p).read()
o='\talpha, err := floatParam(searchParams, "alpha", 0.5)\n'
assert s.count(o)==1
open(p,'w').write(s.replace(o, o+'\tif !linear {\n\t\terr = nil\n\t}\n'))
EOF
go test ./rag -run TestRedisHybridSearchRejectsBadParamsBeforeRedis -v 2>&1 | grep -E -- '--- (FAIL|PASS)|panic'
git checkout -q rag/redis.go && git diff --quiet rag/redis.go && echo restored
```
Expected:
- The LINEAR row passes.
- The RRF row fails, most likely with a nil-client panic that aborts the run.
- The script prints `restored`.

- [ ] **Step 3: Document the behaviour.** Replace the five-line `HybridSearch` doc comment with:

```go
// HybridSearch fuses BM25 text search on Text with vector KNN using FT.HYBRID.
// The query text comes from searchParams["query_text"]; when it is missing or
// matches no document, this falls back to plain KNN (see knnOnly).
// Fusion: searchParams["combine"] = "RRF" (default) or "LINEAR", with optional
// numeric "alpha"/"beta" weights (default 0.5 each). alpha, beta and "ef" are
// validated before any Redis call even when unused, so a bad config fails fast.
// FT.HYBRID takes one vector field, so several fields run one FT.HYBRID each
// (pipelined) and are merged with RRF.
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `gofmt -l rag/ | grep redis; REDIS_ADDR=localhost:6379 go test -tags=integration -race -count=1 ./rag`
Expected: no gofmt output for the Redis files, then `ok`.

- [ ] **Step 5: Commit**

```bash
git add rag/redis.go rag/redis_test.go
git commit -m "test(rag): pin fail-fast validation of unused weights and ef; document it"
```

---

### Task 3: Move the option test to `rag_test.go`, then run the end-to-end check

**Files:**
- Rename: `contextual_rag_test.go` becomes `rag_test.go`.
- Create: a new `contextual_rag_test.go`.

- [ ] **Step 1: Move the option test.** `git mv contextual_rag_test.go rag_test.go` keeps `TestRedisOptions` exactly as it is, in the file named after `rag.go`. Then cut its last check, the four lines from `d := DefaultContextualConfig()` through the closing `}` of that `if`, and put it in a new `contextual_rag_test.go`:

```go
package raggo

import "testing"

func TestDefaultContextualConfig(t *testing.T) {
	d := DefaultContextualConfig()
	if d.DBType != "milvus" || d.DBAddress != "localhost:19530" {
		t.Errorf("contextual defaults changed: %s %s", d.DBType, d.DBAddress)
	}
}
```

- [ ] **Step 2: Confirm no check was lost in the move**

Run: `go test -count=1 -v . -run 'TestRedisOptions|TestDefaultContextualConfig' | grep -- '--- PASS'`
Expected: both `--- PASS: TestRedisOptions` and `--- PASS: TestDefaultContextualConfig`.

- [ ] **Step 3: Run the unit and Redis integration tests**

```bash
go build . ./rag/... ./config/... && go vet . ./rag/... ./config/... && go vet -tags=integration . ./rag/... ./config/...
go test -race -count=1 . ./rag/...
REDIS_ADDR=localhost:6379 go test -tags=integration -race -count=1 ./rag/...
```
Expected: clean and `ok`. The Gemini end-to-end run happens in Task 5, at the very end.

- [ ] **Step 4: Commit**

```bash
git add rag_test.go contextual_rag_test.go
git commit -m "test: move RAG option test to rag_test.go (named after rag.go)"
```

---

### Task 4: USAGE.md for local RAG with llama.cpp and KoboldCpp, backed by a runnable example

Written with the `cc-skills-golang:golang-documentation` skill. raggo is a library, so the documentation leads with working code. Every snippet in USAGE.md is taken from `examples/local_llm/main.go`, which is built and run against both servers in this task, so the doc can't drift from the code.

**Files:**
- Create: `examples/local_llm/main.go`. It lives in its own directory, so `go build ./examples/local_llm` works despite the clashing loose files in `examples/`.
- Create: `USAGE.md` at the repo root.
- Modify: `README.md`, adding one line under "## Examples" that links USAGE.md.

**Why building blocks rather than `raggo.RAG`:** checked in the code.
- `RAG`'s schema hardcodes `Dimension: 1536`, while local embedding models are usually 384–1024 dimensions (embeddinggemma-300M has 768). Every insert would fail the dimension check.
- raggo's LLM calls go through gollm v0.1.1, whose OpenAI provider hardcodes `api.openai.com`.

So the example chunks, embeds, stores and retrieves with raggo, and calls the local chat endpoint directly. Both llama.cpp and KoboldCpp serve OpenAI-compatible `/v1/embeddings` and `/v1/chat/completions`.

**Interfaces:**
- Consumes raggo's public API only:
  - `NewEmbedder`, `SetEmbedderProvider`, `SetEmbedderModel`, `SetEmbedderAPIKey` and `SetOption("api_url", …)`. The OpenAI embedder accepts `api_url` (`rag/providers/openai.go`).
  - `NewVectorDB`, `WithType`, `WithAddress`, `HasCollection`, `DropCollection`, `CreateCollection`, `CreateIndex`, `Insert`, `SetColumnNames` and `HybridSearch`.
  - `NewParser().Parse` (which returns a `Document` with `Content`), plus `NewChunker`, `ChunkSize` and `ChunkOverlap` (`Chunk(text)` returns `[]Chunk` with `.Text`).

- [ ] **Step 1: Write the example program**

`examples/local_llm/main.go`:

```go
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
	if err := run(ctx, *docs, *question,
		env("REDIS_ADDR", "localhost:6379"),
		env("EMBED_URL", "http://localhost:8081/v1/embeddings"),
		env("CHAT_URL", "http://localhost:8080/v1/chat/completions"),
	); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, docsDir, question, redisAddr, embedURL, chatURL string) error {
	// "openai" means any OpenAI-compatible server. Local servers ignore the model
	// name and key, but raggo requires a non-empty key.
	embedder, err := raggo.NewEmbedder(
		raggo.SetEmbedderProvider("openai"),
		raggo.SetEmbedderModel("local"),
		raggo.SetEmbedderAPIKey("none"),
		raggo.SetOption("api_url", embedURL),
	)
	if err != nil {
		return err
	}
	// Size the index from the model instead of assuming 1536.
	probe, err := embedder.Embed(ctx, "dimension probe")
	if err != nil {
		return fmt.Errorf("embeddings endpoint %s: %w", embedURL, err)
	}

	db, err := raggo.NewVectorDB(raggo.WithType("redis"), raggo.WithAddress(redisAddr))
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

	answer, err := chat(ctx, chatURL, question, results)
	if err != nil {
		return err
	}
	fmt.Println("Answer:", answer)
	return nil
}

// index recreates the collection sized to the embedding model, then parses,
// chunks, embeds and stores every .txt and .pdf file in dir.
func index(ctx context.Context, db *raggo.VectorDB, embedder raggo.Embedder, dir string, dim int) error {
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
	fmt.Printf("Indexed %d chunks (%d-dim embeddings)\n", len(records), dim)
	return db.Insert(ctx, collection, records)
}

// chat asks the local model to answer from the retrieved chunks only.
func chat(ctx context.Context, url, question string, results []raggo.SearchResult) (string, error) {
	var sb strings.Builder
	for i, r := range results {
		fmt.Fprintf(&sb, "[%d] %v\n\n", i+1, r.Fields["Text"])
	}
	body, err := json.Marshal(map[string]interface{}{
		"messages": []map[string]string{
			{"role": "system", "content": "Answer using only the provided context. If it does not contain the answer, say you don't know."},
			{"role": "user", "content": "Context:\n" + sb.String() + "Question: " + question},
		},
		"temperature": 0.2,
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
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
```

Run: `gofmt -l examples/local_llm; go build ./examples/local_llm && go vet ./examples/local_llm`
Expected: no gofmt output, and a clean build and vet.

- [ ] **Step 2: Run it against llama.cpp** (`llama-server` is installed at `/home/linuxbrew/.linuxbrew/bin/llama-server`). Start two instances as background processes: the `--embeddings` flag restricts a server to embeddings only, so chat needs its own instance.

```bash
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --port 8081   # background
llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --port 8080                             # background
until curl -sf localhost:8081/health && curl -sf localhost:8080/health; do sleep 2; done
REDIS_ADDR=localhost:6379 EMBED_URL=http://localhost:8081/v1/embeddings CHAT_URL=http://localhost:8080/v1/chat/completions \
  go run ./examples/local_llm
```
Expected:
- The output starts with `Indexed N chunks (768-dim embeddings)`, which confirms the dimension was probed rather than assumed to be 1536.
- `Sources:` lists `sample.txt` among the top 3.
- The `Answer:` mentions traffic, load or scaling. PressureValve is fictional, so that content can only come from retrieval.

Then stop both servers.

- [ ] **Step 3: Run it against KoboldCpp** (downloaded into the session scratchpad, never into the repo)

```bash
S=/tmp/claude-1000/-home-azilber-github-raggo/f9a1a34d-a512-4300-b37e-04fe04ef5455/scratchpad/kobold && mkdir -p $S && cd $S
gh release view --repo LostRuins/koboldcpp --json tagName,assets -q '.tagName, (.assets[].name)'   # pick the linux-x64 asset
gh release download --repo LostRuins/koboldcpp --pattern 'koboldcpp-linux-x64' --clobber && chmod +x koboldcpp-linux-x64
for repo in ggml-org/gemma-3-1b-it-GGUF ggml-org/embeddinggemma-300M-GGUF; do
  f=$(curl -s https://huggingface.co/api/models/$repo | python3 -c 'import sys,json; print(sorted(s["rfilename"] for s in json.load(sys.stdin)["siblings"] if s["rfilename"].endswith(".gguf"))[0])')
  curl -sL -o "${repo#*/}.gguf" "https://huggingface.co/$repo/resolve/main/$f"
done
./koboldcpp-linux-x64 --model gemma-3-1b-it-GGUF.gguf --embeddingsmodel embeddinggemma-300M-GGUF.gguf --port 5001 --gpulayers 99 --quiet   # background
until curl -sf localhost:5001/v1/models; do sleep 2; done
cd /home/azilber/github/raggo
REDIS_ADDR=localhost:6379 EMBED_URL=http://localhost:5001/v1/embeddings CHAT_URL=http://localhost:5001/v1/chat/completions \
  go run ./examples/local_llm
```
Expected: the same three checks as Step 2, all from a single KoboldCpp process on port 5001. Then stop KoboldCpp.

If `gh release download` matches several assets, choose the plain `koboldcpp-linux-x64` build, which includes CUDA; use `--pattern` with its exact name. If KoboldCpp can't load embeddinggemma as an embeddings model, record that as a finding and ask the user before substituting a different embeddings model. Don't swap models silently.

- [ ] **Step 4: Write USAGE.md.** Follow the golang-documentation writing principles: concise, no marketing words, and no claims the runs didn't show. Every command below is one that Steps 2 and 3 ran successfully.

````markdown
# Local RAG with raggo, Redis, and llama.cpp or KoboldCpp

raggo parses and chunks your documents, embeds them through a local OpenAI-compatible `/v1/embeddings` endpoint, and stores and retrieves them in Redis with hybrid (BM25 + vector) search. A local chat model then answers from the retrieved chunks. Nothing leaves your machine.

The complete program is [`examples/local_llm/main.go`](examples/local_llm/main.go).

## Requirements

- Go 1.27.1+
- Redis 8.4+ (for `FT.HYBRID`): `docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4`
- One of the servers below, plus an embeddings GGUF and a chat GGUF. The examples use `ggml-org/embeddinggemma-300M-GGUF` (768-dim) and `ggml-org/gemma-3-1b-it-GGUF`.

## Start the model server

### llama.cpp

A server started with `--embeddings` serves embeddings only, so run two:

```bash
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --port 8081
llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --port 8080
```

```bash
export EMBED_URL=http://localhost:8081/v1/embeddings
export CHAT_URL=http://localhost:8080/v1/chat/completions
```

### KoboldCpp

One process serves both, on port 5001:

```bash
./koboldcpp-linux-x64 --model gemma-3-1b-it.gguf --embeddingsmodel embeddinggemma-300M.gguf --port 5001 --gpulayers 99
```

```bash
export EMBED_URL=http://localhost:5001/v1/embeddings
export CHAT_URL=http://localhost:5001/v1/chat/completions
```

## Run

```bash
REDIS_ADDR=localhost:6379 go run ./examples/local_llm -docs examples/chat/docs -q "What did the PressureValve system do during Black Friday?"
```

It prints the number of chunks indexed and the embedding size, the top 3 sources with their scores, and the model's answer.

## How it hooks up

**Embeddings from the local server.** raggo's `openai` embedder works with any OpenAI-compatible server when you set `api_url`. Local servers ignore the model and key, but raggo requires a non-empty key:

```go
embedder, err := raggo.NewEmbedder(
	raggo.SetEmbedderProvider("openai"),
	raggo.SetEmbedderModel("local"),
	raggo.SetEmbedderAPIKey("none"),
	raggo.SetOption("api_url", embedURL),
)
```

**Redis index sized to the model.** Embed a probe string and use its length as the vector dimension. Redis silently skips indexing any vector whose size differs from the index, so raggo's `Insert` rejects the mismatch instead:

```go
probe, err := embedder.Embed(ctx, "dimension probe")
// ...
{Name: "Embedding", DataType: "float_vector", Dimension: len(probe)},
```

**Chunk, embed, store:** `raggo.NewParser().Parse(file)`, then `raggo.NewChunker(raggo.ChunkSize(200), raggo.ChunkOverlap(50))`, `embedder.Embed` for each chunk, and `db.Insert` with `Embedding`, `Text` and `Metadata` fields.

**Retrieve:** hybrid search fuses BM25 on `Text` with vector similarity. Pass the raw question as `query_text`:

```go
results, err := db.HybridSearch(ctx, collection, map[string]raggo.Vector{"Embedding": qvec}, 3, "COSINE",
	map[string]interface{}{"query_text": question, "ef": 64}, nil)
```

If no document contains any word of the question, this falls back to vector-only search.

**Answer:** POST the retrieved chunks and the question to `CHAT_URL` (`/v1/chat/completions`) and read `choices[0].message.content`. See `chat` in the example.

## Why not `raggo.RAG` or `raggo.SimpleRAG`?

- `RAG`'s collection schema is fixed at 1536 dimensions. Most local embedding models use 384–1024, so every insert would be rejected.
- raggo's built-in LLM calls go through gollm v0.1.1, which only targets `api.openai.com`.

The building blocks above avoid both limits.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `vector has N dimensions, index expects M` | The collection was created with a different embedding model. The example recreates `local_docs` on every run; in your own code, drop or rename the collection when you change models. |
| `ERR unknown command 'FT.HYBRID'` | Redis is older than 8.4. |
| `embeddings endpoint ...: connection refused` | The server is still loading the model; wait for `/health` (llama.cpp) or `/v1/models` (KoboldCpp). |
````

If Step 2 or 3 showed a different reality, fix USAGE.md to match what actually happened. Examples: a KoboldCpp flag, the exact chunk count, or an error string.

Add under `## Examples` in `README.md`:

```markdown
- Local LLMs (llama.cpp or KoboldCpp) with Redis: [USAGE.md](USAGE.md)
```

- [ ] **Step 5: Commit**

```bash
git add examples/local_llm/main.go USAGE.md README.md
git commit -m "docs: USAGE.md for local RAG with llama.cpp/KoboldCpp and Redis, with runnable example"
```

---

### Task 5: Required final end-to-end test

- [ ] **Step 1: Run the full suite, including the Gemini end-to-end test**

```bash
go build . ./rag/... ./config/... ./examples/local_llm && go vet -tags=integration . ./rag/... ./config/...
REDIS_ADDR=localhost:6379 GEMINI_API_KEY=<user's key, env only> go test -tags=integration -race -count=1 -v . ./rag/...
```
Expected: clean and `ok`, including `TestGeminiRAGEndToEnd` with all 13 subtests PASS, **not SKIP**. This is the required end-to-end check: real Gemini embeddings and generation go through `RAG`, `Retriever` and the Redis backend with their default `"ef": 64`. A SKIP means the key wasn't passed and the step isn't done. Pass the key through the environment only, and never write it to a file in the repo.

- [ ] **Step 2: Stop Redis**

```bash
docker stop raggo-redis
```

---

## Self-review
- **Coverage:** item 1 → Task 2; item 2 → Task 1; item 3 is moot, for the reason given in the Spec; item 4 is won't-fix, by the user's choice; item 5 → Task 3. USAGE.md → Task 4, verified against both llama.cpp and KoboldCpp. The required Gemini end-to-end run → Task 5, last. Review Focus 1–6 are pinned in Tasks 1, 1, 2, 2, 3 and 4.
- **Type consistency:** `efRuntime` returns `(int, error)` at both call sites and in the test, and `hybridArgs(..., ef int, cols)` has one caller.
- **Ponytail cuts:**
  - `TestHybridArgs` and a separate `Search` pre-Redis test: `ef` passes through a single argument, and `Search` computes it before its only Redis call.
  - The `TestEfRuntime` rows that duplicate `TestFloatParam`.
  - The "NaN beta" row.
  - The contrived `int32` break.
  - All test reordering: the integration file wasn't asked for, and the user dropped the unit file after weighing the impact.
  - The Gemini end-to-end run stays **required** at the end, at the user's direction.
- **Behaviour change for the PR:** a zero or fractional `ef` used to become 10 silently. It is now an error. Every in-repo default passes `ef` 64 or 100 as an int, which is unaffected.
