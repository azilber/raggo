# raggo.RAG on Local Embeddings + Redis-First README Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Execution method chosen:** subagent-driven (superpowers:subagent-driven-development), in **one git worktree** on **one branch**, delivered as **one PR**.
> After approval, copy this file to `docs/superpowers/plans/2026-09-24-rag-local-embeddings-readme.md` and commit it with Task 1.

**Goal:** Refocus README.md's Quick Start and Configuration on Redis hybrid search using local models, and make that real by letting `raggo.RAG` reach a local embeddings endpoint and size its index from the model instead of a hardcoded 1536.

**Architecture:** The work is staged per golang-refactoring as separate commits on one branch, opened as one PR at the end:
1. **Characterization tests** pin today's `RAG` collection behaviour (Task 1).
2. **A behaviour-preserving extraction** removes the duplicated schema and index literals (Task 2).
3. **The behaviour change** adds `RAGConfig.EmbedURL` and `RAGConfig.Dimension`, with `SetEmbedURL`, `SetDimension` and a `dimension()` probe (Task 3).
4. **Docs:** a runnable `examples/redis_quickstart` and the README Quick Start and Configuration rewrite, with USAGE.md and CLAUDE.md corrections (Task 4).

Structural and behavioural changes never share a commit. The user chose a single PR, which overrides the skill's PR-level separation, so the PR body labels each commit's category for review.

**Tech Stack:** Go 1.27.1, Redis 8.4 (`FT.HYBRID`), `github.com/redis/go-redis/v9` (tests only), and llama.cpp `llama-server` / KoboldCpp for the local runs.

**Spec:** the requirements agreed in chat, no separate file:
- The README Quick Start and Configuration focus on Redis with hybrid search.
- The Quick Start is based on local servers (the user's choice).
- The custom configuration uses a struct, the same pattern as the old Milvus example, and works with local 768-dim models.

## Global Constraints
- **Worktree and branch:** follow superpowers:using-git-worktrees, which the user has already asked for.
  - Step 0 detection (done during planning): this is a normal checkout (`GIT_DIR == GIT_COMMON`, on `main` at `e534b99`, not a submodule), so a worktree is needed.
  - Use the **native `EnterWorktree` tool**, never `git worktree add`. It creates `.claude/worktrees/<name>` on a new branch from `origin/main`, which equals `main` at `e534b99`.
  - `.claude/` is **not** git-ignored, so before creating the worktree, exclude it locally with `.git/info/exclude`. That changes no tracked file.
  - After entering, make sure the branch is `rag-local-embeddings-readme`, renaming it with `git branch -m` if needed.
  - All tasks commit there. The single PR goes to `gh pr create --repo azilber/raggo --base main` at the end of Task 4. Afterwards, `ExitWorktree` with `action: "keep"` until the user merges.
  - All paths in this plan are relative to the worktree root, and the tests use `examples/chat/docs` from there. Every subagent prompt states the worktree's absolute path and says never to touch the main checkout at `/home/azilber/github/raggo`.
- **Clean baseline** (golang-refactoring): start every task from a clean, committed state. If a step turns the tests red, revert to the last green commit instead of debugging forward, and commit as soon as a step goes green.
- **Separation:** never mix structural and behavioural changes in one commit (golang-refactoring). The four task commits stay separate.
- **Build and vet:** `go build . ./rag/... ./config/...` (plus the example directories), and `go vet` with and without `-tags=integration`. Never use `./...`.
- **Redis:** tests use `docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4` and `REDIS_ADDR=localhost:6379`.
- **Commits:** every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Keys:** API keys are passed through the environment only, never written to files.
- **Final check:** the required end-to-end set from CLAUDE.md runs at the end of Task 4, before the PR is opened.

## Review Focus
1. **Existing 1536-dim OpenAI setups** must get exactly the same schema as before. Pinned by `TestRAGCharacterize*` staying unchanged and green through the behaviour change (Task 3).
2. **An unreachable embeddings endpoint when a collection is created** must return an error naming the probe, `measure embedding dimension`, and must not create an index. Pinned by the "unreachable endpoint" row of `TestRAGDimension` (Task 3).
3. **An embedder that returns an empty vector** must be an error, never a 0-dim index. Pinned by the "empty vector" row of `TestRAGDimension` (Task 3).
4. **`ProcessWithContext`, the `ContextualRAG` path,** must size its collection the same way as `LoadDocuments`. Pinned by `TestRAGCharacterizeProcessWithContextSchema` (Tasks 1 and 3).
5. **A local 768-dim model through `raggo.RAG`** must index and return the right chunk with hybrid search on Redis. Pinned by `TestRAGLocalEmbeddingsOnRedis` (Task 3) and by the real llama.cpp and KoboldCpp Quick Start runs (Task 4).

---

## Commits 1–2: safety net, then structure (no behaviour change)

### Task 1: Characterization tests for `RAG` collection creation

**Files:**
- Modify: `rag_integration_test.go`. Add imports and helpers, generalize `dropCollection`, and add two tests. The file is package `raggo_test` behind the `integration` tag.

**Interfaces:**
- Produces (test helpers, used again in Task 3):
  - `func bowVector(text string, dim int) []float64`
  - `type bowEmbedder struct{ dim int }` (implements `providers.Embedder`)
  - `func indexDim(t *testing.T, addr, col string) int64`
  - `func dropCollection(t *testing.T, addr, col string)`
  - `func redisAddrOrSkip(t *testing.T) string`
  - `const pvQuestion`

- [ ] **Step 0: Set up**

From the main checkout, exclude the worktree directory locally (no tracked change), then create the worktree with the native tool:

```bash
cd /home/azilber/github/raggo && git pull --ff-only origin main
grep -qx '.claude/worktrees/' .git/info/exclude || echo '.claude/worktrees/' >> .git/info/exclude
git check-ignore -q .claude/worktrees && echo "worktrees ignored"
```

Expected output: `worktrees ignored`.

Then call **`EnterWorktree`** with `name: "rag-local-embeddings-readme"`. From the new worktree root:

```bash
[ "$(git branch --show-current)" = rag-local-embeddings-readme ] || git branch -m rag-local-embeddings-readme
git log --oneline -1          # expect e534b99 (origin/main)
go mod download
go build . ./rag/... ./config/... ./examples/local_llm && go test -race -count=1 . ./rag/...   # baseline must be green
mkdir -p docs/superpowers/plans && cp /home/azilber/.claude/plans/buzzing-brewing-valiant.md docs/superpowers/plans/2026-09-24-rag-local-embeddings-readme.md
docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4
REDIS_ADDR=localhost:6379 go test -tags=integration -race -count=1 ./rag   # baseline integration must be green
go install golang.org/x/tools/gopls@latest
gopls references rag.go:$(grep -n '^func (r \*RAG) ensureCollection' rag.go | cut -d: -f1):17
gopls call_hierarchy rag.go:$(grep -n '^func (r \*RAG) ProcessWithContext' rag.go | cut -d: -f1):17
```
Expected: the gopls output matches the grep-based blast radius:
- `ensureCollection` has one caller, `LoadDocuments`.
- `ProcessWithContext` is called from `contextual.go`, `contextual_rag.go` (×2) and `examples/chat/v3.go`.

If gopls shows more callers, add them to the ledger before continuing.

- [ ] **Step 1: Generalize the drop helper.** In `rag_integration_test.go`, change the signature to `func dropCollection(t *testing.T, addr, col string)` and replace both uses of `itCollection` inside it with `col`. Update the two Gemini-test call sites:

```go
	dropCollection(t, addr, itCollection)
	t.Cleanup(func() { dropCollection(t, addr, itCollection) })
```

- [ ] **Step 2: Add the helpers and the two characterization tests.** Add these imports: `"hash/fnv"`, `"math"`, `"unicode"` and `"github.com/redis/go-redis/v9"`. Then append:

```go
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
```

- [ ] **Step 3: Run them against the unchanged code, and confirm they catch a change**

Run: `REDIS_ADDR=localhost:6379 go test -tags=integration -race -count=1 -run 'TestRAGCharacterize' -v .`
Expected: PASS for both. Characterization tests describe existing behaviour, so they go green right away.

To prove they have teeth, break each site separately and restore it with `git checkout rag.go` after each:
- Change `Dimension: 1536` to `768` in `ensureCollection`, re-run, and confirm `TestRAGCharacterizeLoadAndQuery` fails with `index dim = 768`.
- Make the same change in `ProcessWithContext`, and confirm `TestRAGCharacterizeProcessWithContextSchema` fails the same way.

If Query's top result isn't PressureValve on unchanged code, stop and investigate; don't loosen the assertion.

- [ ] **Step 4: Commit (tests only)**

```bash
git add rag_integration_test.go docs/superpowers/plans/2026-09-24-rag-local-embeddings-readme.md
git commit -m "test: characterize RAG collection creation (schema dim, hybrid query)"
```

### Task 2: Extract the duplicated schema and index literals (structural)

**Files:**
- Modify: `rag.go`, at `ensureCollection` (~l.714-752) and `ProcessWithContext` (~l.515-542). Add two helpers right after `ensureCollection`.

**Interfaces:**
- Produces:
  - `func collectionSchema(name string, dim int) Schema`
  - `func (r *RAG) vectorIndex() Index`

- [ ] **Step 1: Add the helpers after `ensureCollection`**

```go
// collectionSchema is the schema RAG stores chunks in: an auto ID, the
// embedding, the chunk text and its JSON metadata.
func collectionSchema(name string, dim int) Schema {
	return Schema{
		Name: name,
		Fields: []Field{
			{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
			{Name: "Embedding", DataType: "float_vector", Dimension: dim},
			{Name: "Text", DataType: "varchar", MaxLength: 65535},
			{Name: "Metadata", DataType: "varchar", MaxLength: 65535},
		},
	}
}

// vectorIndex is the HNSW index RAG builds on the Embedding field.
func (r *RAG) vectorIndex() Index {
	return Index{
		Type:   r.config.IndexType,
		Metric: r.config.IndexMetric,
		Parameters: map[string]interface{}{
			"M":              16,
			"efConstruction": 256,
		},
	}
}
```

- [ ] **Step 2: Use them at both sites.** Change only the literals, and keep each site's error handling exactly as it is.
  - In `ensureCollection`, replace the `schema := Schema{...}` literal with `schema := collectionSchema(r.config.Collection, 1536)` and the `index := Index{...}` literal with `index := r.vectorIndex()`.
  - In `ProcessWithContext`, make the same two replacements, keeping its `// Create collection with schema` / `// Create index` comments and its `fmt.Errorf("failed to create collection: %w", err)`-style wrapping.

- [ ] **Step 3: Verify behaviour is preserved**

```bash
gofmt -l rag.go
go build . ./rag/... ./config/... ./examples/local_llm && go vet . ./rag/... ./config/... && go vet -tags=integration . ./rag/... ./config/...
go test -race -count=1 . ./rag/...
REDIS_ADDR=localhost:6379 go test -tags=integration -race -count=1 -run 'TestRAGCharacterize|TestRedis' -v . ./rag/...
git diff --stat
```
Expected: no gofmt output; clean build and vet; everything PASS. The diff touches only `rag.go` and is net-negative in lines. Only literals moved; error strings are unchanged.

- [ ] **Step 4: Commit (structural only)**

```bash
git add rag.go
git commit -m "refactor(rag): extract collectionSchema and vectorIndex from duplicated literals"
```

---

## Commit 3: behaviour, local embeddings endpoint and measured dimension

### Task 3: `EmbedURL`, `Dimension`, `SetEmbedURL`, `SetDimension` and the `dimension()` probe

**Files:**
- Modify: `rag.go`:
  - `RAGConfig` (~l.59-91): two fields.
  - Options: next to `SetDBType` (~l.232).
  - `initialize` (~l.397).
  - A new `dimension` method after `vectorIndex`.
  - The two `collectionSchema(..., 1536)` call sites.
- Test: `rag_test.go` (unit, package `raggo`), and `rag_integration_test.go` (the integration tag).

**Interfaces:**
- Consumes: `collectionSchema` and `vectorIndex` (Task 2); `bowVector`, `dropCollection`, `indexDim`, `redisAddrOrSkip` and `pvQuestion` (Task 1).
- Produces:
  - `RAGConfig.EmbedURL string` and `RAGConfig.Dimension int`
  - `func SetEmbedURL(url string) RAGOption`
  - `func SetDimension(n int) RAGOption`
  - `func (r *RAG) dimension(ctx context.Context) (int, error)`

- [ ] **Step 1: Write the failing unit test.** Add to `rag_test.go`, and extend its imports to `"encoding/json"`, `"net/http"`, `"net/http/httptest"`, `"strings"`, `"sync/atomic"` and `"testing"`:

```go
// embeddingsServer serves an OpenAI-compatible /v1/embeddings that returns
// dim-sized vectors (dim 0 returns an empty vector) and counts requests.
func embeddingsServer(t *testing.T, dim int) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		vec := make([]float64, dim)
		if dim > 0 {
			vec[0] = 1
		}
		json.NewEncoder(w).Encode(map[string]interface{}{
			"data": []map[string]interface{}{{"embedding": vec}},
		})
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func TestRAGDimension(t *testing.T) {
	tests := []struct {
		name      string
		serverDim int
		closed    bool // stop the server before probing
		opts      []RAGOption
		want      int
		wantCalls int32
		wantErr   string
	}{
		{name: "measured from the embedder", serverDim: 768, want: 768, wantCalls: 1},
		{name: "fixed by SetDimension", serverDim: 768, opts: []RAGOption{SetDimension(384)}, want: 384, wantCalls: 0},
		{name: "empty vector is an error", serverDim: 0, wantErr: "empty vector", wantCalls: 1},
		{name: "unreachable endpoint is an error", serverDim: 768, closed: true, wantErr: "measure embedding dimension"},
		{name: "negative Dimension is an error", serverDim: 768, opts: []RAGOption{SetDimension(-1)}, wantErr: "Dimension must be", wantCalls: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv, calls := embeddingsServer(t, tt.serverDim)
			opts := append([]RAGOption{
				SetDBType("memory"),
				SetEmbedURL(srv.URL + "/v1/embeddings"),
				SetAPIKey("none"),
			}, tt.opts...)
			r, err := NewRAG(opts...)
			if err != nil {
				t.Fatal(err)
			}
			defer r.Close()
			if tt.closed {
				srv.Close()
			}
			got, err := r.dimension(t.Context())
			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("err = %v, want it to contain %q", err, tt.wantErr)
				}
			} else if err != nil || got != tt.want {
				t.Errorf("dimension() = %d, %v; want %d, nil", got, err, tt.want)
			}
			if n := calls.Load(); n != tt.wantCalls {
				t.Errorf("embeddings requests = %d, want %d", n, tt.wantCalls)
			}
		})
	}
}
```

- [ ] **Step 2: Write the failing integration test.** Add to `rag_integration_test.go`, with the imports `"net/http/httptest"` and `"sync/atomic"`:

```go
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
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `go test ./ -run TestRAGDimension -v`
Expected: FAIL to compile with `undefined: SetEmbedURL`, `undefined: SetDimension` and `r.dimension undefined`.

- [ ] **Step 4: Implement.** In `RAGConfig`, add after `APIKey`:

```go
	EmbedURL  string // OpenAI-compatible embeddings endpoint; empty uses the provider's default
	Dimension int    // Embedding size for new collections; 0 measures it from the embedder
```

Add after `SetDBType`:

```go
// SetEmbedURL points the embedder at an OpenAI-compatible embeddings endpoint,
// such as a local llama.cpp or KoboldCpp server
// ("http://localhost:8081/v1/embeddings"). The API key is sent to this URL as a
// Bearer token, so only use endpoints you trust.
func SetEmbedURL(url string) RAGOption {
	return func(c *RAGConfig) {
		c.EmbedURL = url
	}
}

// SetDimension fixes the embedding size used when RAG creates a collection.
// Leave it at 0 (the default) to measure it from the embedder, which costs one
// embedding request per collection created. Negative values are an error.
func SetDimension(n int) RAGOption {
	return func(c *RAGConfig) {
		c.Dimension = n
	}
}
```

In `initialize`, replace the `embedder, err := NewEmbedder(...)` call with:

```go
	embedOpts := []EmbedderOption{
		SetEmbedderProvider(r.config.Provider),
		SetEmbedderModel(r.config.Model),
		SetEmbedderAPIKey(r.config.APIKey),
	}
	if r.config.EmbedURL != "" {
		embedOpts = append(embedOpts, SetOption("api_url", r.config.EmbedURL))
	}
	embedder, err := NewEmbedder(embedOpts...)
```

Add after `vectorIndex`:

```go
// dimension returns the embedding size for a new collection: the configured
// Dimension, or, when that is 0, the length of one probe embedding. The result
// is not stored, so concurrent callers share no state.
func (r *RAG) dimension(ctx context.Context) (int, error) {
	if r.config.Dimension < 0 {
		return 0, fmt.Errorf("RAGConfig.Dimension must be >= 0 (0 measures it), got %d", r.config.Dimension)
	}
	if r.config.Dimension > 0 {
		return r.config.Dimension, nil
	}
	vec, err := r.embedder.Embed(ctx, "dimension probe")
	if err != nil {
		return 0, fmt.Errorf("measure embedding dimension: %w", err)
	}
	if len(vec) == 0 {
		return 0, fmt.Errorf("measure embedding dimension: embedder returned an empty vector")
	}
	return len(vec), nil
}
```

In `ensureCollection`, replace `schema := collectionSchema(r.config.Collection, 1536)` with:

```go
		dim, err := r.dimension(ctx)
		if err != nil {
			return err
		}
		schema := collectionSchema(r.config.Collection, dim)
```

In `ProcessWithContext`, replace `schema := collectionSchema(r.config.Collection, 1536)` with:

```go
		dim, err := r.dimension(ctx)
		if err != nil {
			return fmt.Errorf("failed to create collection: %w", err)
		}
		schema := collectionSchema(r.config.Collection, dim)
```

- [ ] **Step 5: Run the tests and confirm they pass, including the untouched characterization tests**

```bash
gofmt -l rag.go rag_test.go rag_integration_test.go
go build . ./rag/... ./config/... ./examples/local_llm && go vet . ./rag/... ./config/... && go vet -tags=integration . ./rag/... ./config/...
go test -race -count=1 -run TestRAGDimension -v .
REDIS_ADDR=localhost:6379 go test -tags=integration -race -count=1 -run 'TestRAGCharacterize|TestRAGLocalEmbeddingsOnRedis|TestRedis' -v . ./rag/...
```
Expected:
- `TestRAGDimension` passes all 5 subtests.
- `TestRAGLocalEmbeddingsOnRedis` passes (index dim 768, PressureValve on top).
- Both `TestRAGCharacterize*` still pass with no edits, which means existing 1536-dim setups are unchanged.

- [ ] **Step 6: Security and safety pass (golang-refactoring requires it for a behavioural step).** Check these three things and record them in the PR body:
  - `EmbedURL` receives the API key as a Bearer token, which the doc comment warns about.
  - The probe sends only the fixed string `"dimension probe"`.
  - `dimension()` writes no shared state, so there's no data race; `go test -race` above covers it.

- [ ] **Step 7: Commit (behavioural)**

```bash
git add rag.go rag_test.go rag_integration_test.go
git commit -m "feat(rag): SetEmbedURL and measured Dimension so RAG works with local embedding models"
```

---

## Commit 4: README, USAGE.md, CLAUDE.md and a runnable Quick Start

### Task 4: `examples/redis_quickstart` and the README Quick Start and Configuration

**Files:**
- Create: `examples/redis_quickstart/main.go`
- Modify: `README.md`. Change the summary line, `## Quick Start` (l.14-45) and `## Configuration` (l.47-88); leave the rest alone.
- Modify: `USAGE.md`. Change the l.163 sentence and replace the `## Why not raggo.RAG with llama.cpp or KoboldCpp?` section (l.190-195).
- Modify: `CLAUDE.md`, at l.7 (build line), l.86, l.104 and l.108.

**Interfaces:**
- Consumes: `SetEmbedURL` and `SetDimension`, and the `EmbedURL` and `Dimension` fields (Task 3).

- [ ] **Step 0: Start the servers** (in the same worktree)

```bash
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --device none --port 8081 &
until curl -sf localhost:8081/health; do sleep 2; done
```

- [ ] **Step 1: Write the example.** `examples/redis_quickstart/main.go`:

```go
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
```

- [ ] **Step 2: Run it against llama.cpp on CPU and capture the output**

```bash
S=/tmp/claude-1000/-home-azilber-github-raggo/f9a1a34d-a512-4300-b37e-04fe04ef5455/scratchpad
gofmt -l examples/redis_quickstart; go vet ./examples/redis_quickstart
docker exec raggo-redis redis-cli FT.DROPINDEX quickstart DD >/dev/null 2>&1; go run ./examples/redis_quickstart 2>&1 | tee $S/quickstart-llamacpp.txt
docker exec raggo-redis redis-cli FT.INFO quickstart | grep -A1 -x dim
```
Expected:
- The first printed result line mentions PressureValve.
- The `FT.INFO` output shows `dim` 768.
- The output includes raggo's own log lines (such as `Inserting ... records`); keep them, so the README shows the real output.

- [ ] **Step 3: Run it against KoboldCpp on CPU**

```bash
pkill -x llama-server
cd /tmp/koboldcpp && ./koboldcpp-linux-x64 --usecpu --model gemma-3-1b-it-Q4_K_M.gguf --embeddingsmodel embeddinggemma-300M-Q8_0.gguf --port 5001 --quiet &
cd - && until curl -sf localhost:5001/v1/models; do sleep 2; done
docker exec raggo-redis redis-cli FT.DROPINDEX quickstart DD >/dev/null 2>&1; EMBED_URL=http://localhost:5001/v1/embeddings go run ./examples/redis_quickstart 2>&1 | tee $S/quickstart-kobold.txt
pkill -f '^\./koboldcpp-linux-x64'
```
Expected: the first result mentions PressureValve.

- [ ] **Step 4: Rewrite the README summary, Quick Start and Configuration.** Replace the summary blockquote (l.3) with:

```markdown
> A RAG (Retrieval Augmented Generation) library for Go: load, chunk and embed documents, then retrieve them with Redis hybrid (BM25 + vector) search. Milvus, chromem and an in-memory store are also supported.
```

Replace `## Quick Start` through the line before `## Configuration` with the text below. Replace `OUTPUT` with the verbatim contents of `$S/quickstart-llamacpp.txt`. It's a runtime artifact, not a placeholder.

````markdown
## Quick Start

Index a folder into Redis and run a hybrid (BM25 + vector) query, with embeddings from a local model on CPU. You need Go 1.27.1+, Docker, and llama.cpp's `llama-server`. KoboldCpp and Gemini work too; see [USAGE.md](USAGE.md).

```bash
docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --device none --port 8081
```

```go
r, err := raggo.NewRAG(
	raggo.WithRedis("quickstart"),                                // Redis 8.4+ at localhost:6379; hybrid search is on by default
	raggo.SetEmbedURL("http://localhost:8081/v1/embeddings"),     // any OpenAI-compatible embeddings endpoint
	raggo.SetAPIKey("none"),                                      // local servers ignore it, but raggo requires one
)
if err != nil {
	log.Fatal(err)
}
defer r.Close()

if err := r.LoadDocuments(ctx, "examples/chat/docs"); err != nil { // index size is measured from the model (768 here)
	log.Fatal(err)
}
results, err := r.Query(ctx, "What did the PressureValve system do during Black Friday?")
```

The full program is [`examples/redis_quickstart`](examples/redis_quickstart/main.go):

```bash
go run ./examples/redis_quickstart
```

```text
OUTPUT
```

Each line is a retrieved chunk and its score. To generate an answer from those chunks with a local chat model, see [USAGE.md](USAGE.md).
````

Replace `## Configuration` through the line before `## Table of Contents` with:

````markdown
## Configuration

Start from `raggo.DefaultRAGConfig()`, set what you need, and pass it to `raggo.NewRAG`:

```go
cfg := raggo.DefaultRAGConfig()
cfg.DBType = "redis"
cfg.DBAddress = "localhost:6379"                     // or "redis://user:pass@host:6379/0"
cfg.Collection = "my_documents"
cfg.EmbedURL = "http://localhost:8081/v1/embeddings" // llama.cpp; KoboldCpp: http://localhost:5001/v1/embeddings
cfg.APIKey = "none"                                  // local servers ignore it; raggo requires one
cfg.Dimension = 0                                    // 0 = measure from the model (768 for embeddinggemma-300M)
cfg.IndexMetric = "COSINE"
cfg.UseHybrid = true                                 // BM25 on the chunk text + vector KNN, fused by FT.HYBRID
cfg.TopK = 5
cfg.MinScore = 0.5                                   // scores are normalized to [0,1]; higher is better
cfg.ChunkSize = 300
cfg.ChunkOverlap = 50
cfg.SearchParams = map[string]interface{}{
	"combine": "RRF", // or "LINEAR" with "alpha" and "beta" weights
	"ef":      64,    // HNSW search depth
}

r, err := raggo.NewRAG(func(c *raggo.RAGConfig) { *c = *cfg })
```

The same settings as options:

```go
r, err := raggo.NewRAG(
	raggo.SetDBType("redis"),
	raggo.SetDBAddress("localhost:6379"),
	raggo.SetCollection("my_documents"),
	raggo.SetEmbedURL("http://localhost:8081/v1/embeddings"),
	raggo.SetAPIKey("none"),
	raggo.SetSearchStrategy("hybrid"),
	raggo.SetTopK(5),
	raggo.SetMinScore(0.5),
)
```

Redis search parameters (`SearchParams`) are checked before any Redis call, even the ones the chosen fusion doesn't use:

| Key | Values | Default |
|---|---|---|
| `combine` | `RRF` or `LINEAR` | `RRF` |
| `alpha`, `beta` | `LINEAR` weights, numbers ≥ 0 | 0.5 each |
| `ef` | HNSW search depth, a whole number ≥ 1 | 64 in `DefaultRAGConfig` |

`query_text`, the text half of hybrid search, is added automatically from the query. Redis 8.4+ is required (`FT.HYBRID`).

The `config` package (`config.LoadConfig` and the `RAGGO_*` variables) isn't read by the constructors yet; configure through `RAGConfig` or the options above.
````

- [ ] **Step 5: Run the Configuration snippet exactly as written** (as a scratch program, not committed). Start llama.cpp again, then run:

```bash
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --device none --port 8081 &
until curl -sf localhost:8081/health; do sleep 2; done
cat > $S/configcheck.go <<'EOF'
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/teilomillet/raggo"
)

func main() {
	ctx := context.Background()
	cfg := raggo.DefaultRAGConfig()
	cfg.DBType = "redis"
	cfg.DBAddress = "localhost:6379"
	cfg.Collection = "my_documents"
	cfg.EmbedURL = "http://localhost:8081/v1/embeddings"
	cfg.APIKey = "none"
	cfg.Dimension = 0
	cfg.IndexMetric = "COSINE"
	cfg.UseHybrid = true
	cfg.TopK = 5
	cfg.MinScore = 0.5
	cfg.ChunkSize = 300
	cfg.ChunkOverlap = 50
	cfg.SearchParams = map[string]interface{}{"combine": "RRF", "ef": 64}
	r, err := raggo.NewRAG(func(c *raggo.RAGConfig) { *c = *cfg })
	if err != nil {
		log.Fatal(err)
	}
	defer r.Close()
	if err := r.LoadDocuments(ctx, "examples/chat/docs"); err != nil {
		log.Fatal(err)
	}
	res, err := r.Query(ctx, "What did the PressureValve system do during Black Friday?")
	if err != nil {
		log.Fatal(err)
	}
	if len(res) == 0 {
		log.Fatal("no results")
	}
	fmt.Println(len(res), "results; top:", res[0].Score)
}
EOF
printf '{"Replace":{"%s/examples/zz_configcheck/main.go":"%s/configcheck.go"}}' "$PWD" "$S" > $S/ov.json
docker exec raggo-redis redis-cli FT.DROPINDEX my_documents DD >/dev/null 2>&1; go run -overlay $S/ov.json ./examples/zz_configcheck
docker exec raggo-redis redis-cli FT.INFO my_documents | grep -A1 -x dim
pkill -x llama-server
```
Expected: `N results; top: …` with N ≥ 1, and `dim` 768. Fix the README if reality differs.

- [ ] **Step 6: Correct USAGE.md and CLAUDE.md**
  - **USAGE.md l.163:** replace "because Gemini's embeddings API accepts a `dimensions` parameter that matches `RAG`'s fixed 1536-dimension schema. raggo's built-in `openai` embedder can't send that parameter, so register your own provider and point `RAG` at it:" with: "`raggo.RAG` measures the embedding size, so any Gemini dimension works. The registered provider below also asks Gemini for 1536 dimensions, which raggo's built-in `openai` embedder can't request:"
  - **USAGE.md:** replace the whole `## Why not raggo.RAG with llama.cpp or KoboldCpp?` section with:

    ```markdown
    ## `raggo.RAG` with local models

    `raggo.RAG` can index and search with a local embeddings server: `SetEmbedURL` points it at the server and the index is sized from the model (see the README Quick Start and `examples/redis_quickstart`). What it can't do is answer from a local chat model, because raggo's built-in LLM calls use gollm's `openai` provider, whose endpoint is fixed at `api.openai.com` in gollm v0.1.1. `examples/local_llm` makes that chat call directly.
    ```
  - **CLAUDE.md l.7:** add `./examples/redis_quickstart` to the build line.
  - **CLAUDE.md l.86:** replace "`RAG`'s collection schema hardcodes `Dimension: 1536` (`rag.go`)." with "`RAG` sizes new collections from `RAGConfig.Dimension`, or, when that's 0, from one probe embedding (`dimension()` in `rag.go`); `SetEmbedURL` points its embedder at any OpenAI-compatible endpoint."
  - **CLAUDE.md l.104:** replace "(for example Gemini's `dimensions: 1536`, which makes it fit `RAG`'s fixed schema)" with "(for example Gemini's `dimensions: 1536`)".
  - **CLAUDE.md l.108:** replace "The example uses the building blocks rather than `RAG` for two reasons: `RAG`'s schema is fixed at 1536 dimensions while local embedding models are usually smaller, and gollm's `openai` endpoint can't be pointed at a local server." with "`examples/redis_quickstart` shows `raggo.RAG` on a local embeddings server; `examples/local_llm` uses the building blocks because answer generation needs a direct chat call (gollm's `openai` endpoint can't be pointed at a local server)."

- [ ] **Step 7: Build, then run the required end-to-end set from CLAUDE.md**

```bash
go build . ./rag/... ./config/... ./examples/local_llm ./examples/redis_quickstart && go vet . ./rag/... ./config/... && go vet -tags=integration . ./rag/... ./config/...
REDIS_ADDR=localhost:6379 GEMINI_API_KEY=<user's key, env only> go test -tags=integration -race -count=1 -v . ./rag/...
```
Expected: everything passes, including `TestGeminiRAGEndToEnd` 13/13 PASS, not SKIP (it now goes through the `dimension()` probe), `TestRAGCharacterize*` and `TestRAGLocalEmbeddingsOnRedis`.

Then run the llama.cpp and KoboldCpp `local_llm` procedures exactly as written in CLAUDE.md's "End-to-end tests" section, reusing `/tmp/koboldcpp`, each meeting the three pass criteria. Stop all servers and run `docker stop raggo-redis`.

- [ ] **Step 8: Commit (docs), then push and open the single PR.** The final whole-branch review in subagent-driven-development happens before the PR is opened.

```bash
git add examples/redis_quickstart/main.go README.md USAGE.md CLAUDE.md
git commit -m "docs: Redis hybrid Quick Start and RAGConfig Configuration on local models"
git push -u origin rag-local-embeddings-readme
gh pr create --repo azilber/raggo --base main --title "raggo.RAG on local embeddings + Redis hybrid Quick Start and Configuration" --body "Commits, by category (golang-refactoring keeps them separate; one PR by request):
1. test: characterization tests pinning RAG collection creation (1536-dim schema, hybrid query, ProcessWithContext schema).
2. refactor (structural, no behaviour change): extract collectionSchema/vectorIndex from the duplicated literals; same tests green before and after.
3. feat (behavioural): RAGConfig.EmbedURL/SetEmbedURL (OpenAI-compatible endpoint, e.g. llama.cpp/KoboldCpp) and RAGConfig.Dimension/SetDimension (0 = measure with one probe embedding when a collection is created), replacing the hardcoded 1536. The characterization tests pass unchanged, so 1536-dim setups get the same schema; other sizes (local 768-dim, text-embedding-3-large 3072) no longer fail on insert. Cost: one embedding request per new collection without SetDimension. Security/safety: the API key goes to EmbedURL as a Bearer token (documented); the probe sends a fixed string; dimension() shares no state (-race clean).
4. docs: README Quick Start = raggo.RAG + Redis hybrid on a local llama.cpp embeddings server (runnable examples/redis_quickstart; output from a real CPU run, also verified on KoboldCpp --usecpu). Configuration = RAGConfig struct replacing the non-compiling config.Config example, plus options and a Redis searchParams table. USAGE.md/CLAUDE.md corrected.

Test plan: unit + integration suites under -race; required E2E set (Gemini 13/13, llama.cpp and KoboldCpp local_llm) passes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

---

## Skill reviews applied (using-git-worktrees, golang-refactoring, golang-context, golang-safety)
- **Worktrees:**
  - Use the native `EnterWorktree`. Exclude `.claude/worktrees/` through `.git/info/exclude`, because `.claude` isn't ignored and the worktree would otherwise show up as an untracked nested repo.
  - Check the branch name, and require green unit and integration baselines before Task 1.
  - Subagents get the absolute worktree path. Finish with `ExitWorktree` and `keep` until the merge.
- **Refactoring:**
  - Clean-commit baseline per task, reverting rather than debugging forward.
  - The mutation check now covers both collection-creation sites.
  - Tests, structural, behavioural and docs changes are separate commits.
- **Context:**
  - The context already flows `dimension(ctx)` → `EmbeddingService.Embed(ctx)` → `http.NewRequestWithContext` (`rag/providers/openai.go:111`).
  - New tests use `t.Context()`.
  - `NewRAG`'s internal `context.Background()` for `Connect` predates this work and is out of scope.
- **Safety:**
  - A negative `Dimension` is an error, with a test row.
  - An empty probe vector is an error.
  - `dimension()` writes no shared state, so it's race-free.
  - The scratch Configuration check guards `res[0]`.
  - Accepted: the README's `*c = *cfg` shares the caller's `SearchParams` map. The snippet never reuses `cfg`, so a defensive copy would only add noise.

## Self-review
- **Spec coverage:**

  | Requirement | Where |
  |---|---|
  | Redis + hybrid Quick Start on local servers | Task 4, Steps 1–4 |
  | Struct-based custom configuration like the Milvus example | Task 4, Steps 4–5 |
  | Works with local 768-dim models | Task 3 (`dimension()`), proven in Tasks 3 and 4 |
  | golang-refactoring staging | separate commits: tests, structure, behaviour, docs; one PR (the user's choice) |
  | Required end-to-end set | Task 4, Step 7 |

- **Placeholder scan:** `OUTPUT` in Task 4 Step 4 is filled from the captured run file, and the key is supplied at run time. Nothing else is left open.
- **Type consistency:**
  - `collectionSchema(name string, dim int) Schema` and `vectorIndex() Index` (Task 2) are used unchanged in Task 3.
  - The Task 1 helpers (`bowVector`, `indexDim`, `dropCollection(t, addr, col)`, `redisAddrOrSkip`, `pvQuestion`) are reused in Task 3 with the same signatures.
  - `SetEmbedURL`, `SetDimension` and `dimension()` (Task 3) are used in Task 4.
