# Redis Vector + Hybrid Search Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Execution method chosen:** Native (superpowers:executing-plans), with one whole-branch review at the end.
> After approval, copy this file to `docs/superpowers/plans/2026-09-23-redis-vector-backend.md` and commit it with Task 1.

**Goal:** Add Redis (Query Engine, Redis 8.4+) as a `rag.VectorDB` backend.
- It is selected with `WithType("redis")`, the same way as Milvus.
- `Search` uses KNN in `FT.SEARCH`.
- `HybridSearch` uses `FT.HYBRID`, which combines BM25 full-text search on `Text` with a vector search and merges the two with RRF or LINEAR fusion.
- Hybrid search works across several vector fields.
- `ContextualRAG` can be pointed at Redis.

**Architecture:**
- One new file, `rag/redis.go`, implements the existing `rag.VectorDB` interface. The interface does not change.
- Each collection is an FT index over hashes stored at `<collection>:<id>`.
- `CreateCollection` only remembers the schema. `CreateIndex` issues `FT.CREATE`, because Redis defines the vector metric and HNSW parameters in that one command.
- Hybrid search gets the user's query text through `searchParams["query_text"]`, which the two raggo callers now set. Milvus and the memory backend ignore the key.

**Tech Stack:**
- Go 1.23 module (local toolchain is 1.27).
- `github.com/redis/go-redis/v9` (v9.22+), used for the connection, pipelines, `FT.CREATE`, `FT.SEARCH` and `FT.DROPINDEX`.
- `FT.HYBRID` is sent through `client.Do` with the exact documented syntax (https://redis.io/docs/latest/commands/ft.hybrid/). The typed `FTHybridWithArgs` is not used: its vector serialization reuses the vector-set `Vector` type (`VectorFP32.Value()` emits `FP32 <blob>`, the VADD format), which isn't clearly correct for `VSIM`.

**Spec:** the design approved in this session, summarized in the Architecture section above. It went through a ponytail review, and the user asked to include the LINEAR combine option, multi-vector hybrid search and the ContextualRAG Redis option. There is no separate spec file, because this was classified as bounded.

## Global Constraints
- **Redis version:** 8.4.0 or later, the first release with `FT.HYBRID`. Tests use the `redis:8.4` image.
- **Protocol:** the client uses `Protocol: 2`, because go-redis only parses `FT.SEARCH` replies on RESP2.
- **Vector encoding:** `FLOAT32`, little-endian, `4*dim` bytes.
- **Score semantics:** a higher score is better, because callers filter with `result.Score < MinScore` (`retriever.go:162`, `rag.go:819`).
  - Search with COSINE or IP: `1 - dist`.
  - Search with L2: `1/(1+dist)`.
  - RRF hybrid: normalized so that a document ranked first in every list scores `1.0`.
- **RRF constant:** `60`, the same default as `rag/reranker.go`.
- **Safety** (golang-safety review):
  - **Type assertions:** every assertion uses comma-ok, so nothing panics the way `MilvusDB`'s bare `.(int)` does. Unsupported field types return errors instead of `panic`.
  - **Concurrency:** `schemas` and `columnNames` are guarded by `sync.RWMutex`, following `MemoryDB`, because `Insert` is called from goroutines.
  - **Nil maps:** `schemas` is initialized lazily, so a zero-value `RedisDB` can't panic on a nil map.
  - **Slice aliasing:** caller slices (`SetColumnNames`, `Schema.Fields`) are cloned rather than aliased.
  - **Batch writes:** `Insert` uses `TxPipeline`, so a batch is all-or-nothing.
  - **Bad replies:** a missing `__key`/`__score` or an unparseable `__dist` returns an error rather than a silent `ID 0` or `Score 0`.
  - **Arguments:** `topK <= 0` returns an error.
  - **Float comparisons:** tests compare against `1+1e-9` rather than exactly 1.
  - **Accepted limits:** calling a method before `Connect` nil-derefs `client`, the same as `MilvusDB`. A non-`float64` `alpha`/`beta` or a non-`int` `ef` silently uses the default.
- **Database** (golang-database review):
  - **Parameterized queries:** vectors, `K` and `EF` go through `PARAMS`, and query text is reduced to letters and digits. Index and field names can't be parameterized, so they are checked against `^[A-Za-z_][A-Za-z0-9_]*$`.
  - **Connection pool:** `Config.MaxPoolSize` sets `PoolSize`.
  - **Batched inserts:** `Insert` validates everything first, then writes in `MULTI/EXEC` batches of 500. Each batch is atomic. If a later batch fails, the error says earlier batches were written.
  - **Index design:** only the vector fields and `Text` are indexed. `Metadata` is stored in the hash but not indexed.
  - **Error wrapping:** go-redis errors are wrapped with the command and collection name.
  - **Not applicable:** row closing, `sql.ErrNoRows` and migrations have no Redis equivalent. The index schema is defined by the caller, the same as with Milvus.
- **Context:** every Redis call takes the caller's `ctx`. There is no `context.Background()` outside tests and no ctx stored in a struct. `ContextTimeoutEnabled: true`. The existing `context.Background()` in `contextual_rag.go` `initializeRAG` is left alone because it predates this change.
- **Build rule:** don't run `go build ./...` (the `examples/` mains clash). Use `go build . ./rag/... ./config/...`.
- **Commits:** each commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus
1. **A password in a `redis://` URL.** Connection errors must not echo the password. It is pinned by `TestRedisConnectErrorHidesPassword` in Task 1.
2. **A vector whose length doesn't match the index `DIM`.** Redis silently leaves that hash out of the index, so the data is lost for search. `Insert` must return an error instead. Pinned in Task 2's integration test.
3. **User text containing Redis query-syntax characters** (`?`, `-`, `@`, `"`, `|`). `FT.HYBRID` must not fail with a syntax error. Pinned by `TestTextQuery` in Task 1 and the `"sourdough bread?"` query in Task 3.
4. **The RAG defaults (`UseHybrid: true`, `MinScore: 0.7`) on Redis.** Fused scores must be in `[0,1]`, with the best document near 1, rather than raw RRF values around `0.03` that `MinScore` would filter out entirely. Pinned in Task 3's score assert. Known limit: a document found by only one half of a hybrid query scores at most `0.5`.
6. **A caller cancels the context or its deadline passes.** go-redis ignores `ctx` on socket I/O unless `ContextTimeoutEnabled` is set, so without it a cancelled request would keep running until the fixed 3-second `ReadTimeout`. `Connect` sets it, and a positive `Config.Timeout` raises the socket timeouts for slow `FT.CREATE` and `FT.HYBRID` calls. Pinned in Task 2 (`Search` with a cancelled ctx returns `context.Canceled`).
7. **Concurrent `Insert`, `CreateCollection` and `SetColumnNames`,** as `concurrentloader.go` and the examples do. These must not race or crash with a concurrent map write. Pinned in Task 2 under `-race`.
5. **Restarting the process against an existing index.** A new `RedisDB` hasn't seen `CreateCollection`, yet `Insert`, `CreateIndex` (a no-op) and `Search` must still work. Pinned in Task 2's second-instance step.

---

### Task 1: Dependency and pure helpers

**Files:**
- Create: `rag/redis.go`
- Create: `rag/redis_test.go`
- Modify: `go.mod`, `go.sum`

**Interfaces:**
- Produces, all in package `rag` and unexported:
  - `const rrfK = 60.0`
  - `func float32Bytes(v []float64) []byte`
  - `func redisValue(v interface{}) (interface{}, error)`
  - `func redisMetric(metric string) string`
  - `func distToScore(dist float64, metric string) float64`
  - `func textQuery(text string) string`
  - `func fuseRRF(lists [][]SearchResult, topK int) []SearchResult`
  - `func keyID(key string) int64`
  - `func efRuntime(params map[string]interface{}) int`
  - `type RedisDB struct`, `func newRedisDB(cfg *Config) (*RedisDB, error)`
  - `func (r *RedisDB) Connect(ctx context.Context) error`
  - `func (r *RedisDB) Close() error`

- [ ] **Step 1: Add the dependency**

Run: `go get github.com/redis/go-redis/v9@latest`
Expected: `go.mod` gains `github.com/redis/go-redis/v9 v9.2x.x`. The `go` line may be raised to go-redis's minimum, which is fine.

- [ ] **Step 2: Write the failing tests**

`rag/redis_test.go`:

```go
package rag

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestTextQuery(t *testing.T) {
	if got := textQuery(`what is "redis" @vector -search?`); got != "@Text:(what|is|redis|vector|search)" {
		t.Errorf("textQuery = %q", got)
	}
	if got := textQuery(" ?!-@ "); got != "*" {
		t.Errorf("textQuery of punctuation = %q, want *", got)
	}
}

func TestValidName(t *testing.T) {
	if err := validName("field", "Embedding"); err != nil {
		t.Error(err)
	}
	for _, bad := range []string{"", "a b", "x]=>[KNN", "@f", "col*"} {
		if validName("field", bad) == nil {
			t.Errorf("validName accepted %q", bad)
		}
	}
}

func TestFuseRRF(t *testing.T) {
	a := []SearchResult{{ID: 1}, {ID: 2}}
	b := []SearchResult{{ID: 2}, {ID: 3}}
	got := fuseRRF([][]SearchResult{a, b}, 2)
	if len(got) != 2 || got[0].ID != 2 {
		t.Fatalf("fuseRRF = %+v, want doc 2 (in both lists) first, 2 results", got)
	}
	if got[0].Score <= 0 || got[0].Score > 1+1e-9 {
		t.Errorf("fused score %v not in (0,1]", got[0].Score)
	}
}

func TestRedisConnectErrorHidesPassword(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	db, _ := newRedisDB(&Config{Address: "redis://user:s3cret@127.0.0.1:1/0"})
	err := db.Connect(ctx)
	if err == nil {
		t.Fatal("expected connection error on port 1")
	}
	if strings.Contains(err.Error(), "s3cret") {
		t.Errorf("error leaks password: %v", err)
	}
}
```

- [ ] **Step 3: Run the tests and confirm they fail**

Run: `go test ./rag -run 'TestTextQuery|TestValidName|TestFuseRRF|TestRedisConnectErrorHidesPassword' -v`
Expected: FAIL to compile with `undefined: textQuery`.

- [ ] **Step 4: Write the implementation**

`rag/redis.go`:

```go
// Package rag provides retrieval-augmented generation capabilities.
package rag

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"unicode"

	"github.com/redis/go-redis/v9"
)

// rrfK is the Reciprocal Rank Fusion constant, matching RRFReranker's default.
const rrfK = 60.0

// RedisDB implements VectorDB on the Redis Query Engine (Redis 8.4+ for FT.HYBRID).
// Each collection is an FT index over hashes stored at "<collection>:<id>".
type RedisDB struct {
	client      *redis.Client
	config      *Config
	mu          sync.RWMutex      // guards columnNames and schemas, as in MemoryDB (callers Insert from goroutines)
	columnNames []string
	schemas     map[string]Schema // FT.CREATE needs the metric, which only arrives with CreateIndex
}

// columns returns the current result columns. SetColumnNames swaps in a fresh
// slice rather than mutating, so the returned slice is safe to read unlocked.
func (r *RedisDB) columns() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.columnNames
}

// schema returns the schema recorded by CreateCollection, if this process saw one.
func (r *RedisDB) schema(name string) (Schema, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.schemas[name]
	return s, ok
}

// newRedisDB creates a new RedisDB instance with the given configuration.
// Note: This doesn't establish the connection - call Connect() separately.
func newRedisDB(cfg *Config) (*RedisDB, error) {
	return &RedisDB{config: cfg, schemas: make(map[string]Schema)}, nil
}

// Connect accepts "host:port" or a redis:// / rediss:// URL (for password and DB).
// Errors name only host:port so credentials in the URL never reach logs.
func (r *RedisDB) Connect(ctx context.Context) error {
	opts := &redis.Options{Addr: r.config.Address}
	if strings.HasPrefix(r.config.Address, "redis://") || strings.HasPrefix(r.config.Address, "rediss://") {
		var err error
		if opts, err = redis.ParseURL(r.config.Address); err != nil {
			return fmt.Errorf("invalid Redis URL") // err would echo the URL, password included
		}
	}
	opts.Protocol = 2               // go-redis parses FT.SEARCH replies only on RESP2
	opts.ContextTimeoutEnabled = true // honor caller ctx deadlines/cancel on socket I/O (off by default)
	if r.config.Timeout > 0 {
		opts.ReadTimeout, opts.WriteTimeout = r.config.Timeout, r.config.Timeout // FT.CREATE/FT.HYBRID can outlast the 3s default
	}
	if r.config.MaxPoolSize > 0 {
		opts.PoolSize = r.config.MaxPoolSize // otherwise go-redis default: 10 per CPU
	}
	r.client = redis.NewClient(opts)
	if err := r.client.Ping(ctx).Err(); err != nil {
		return fmt.Errorf("failed to connect to Redis at %s: %w\nPlease ensure Redis 8.4+ is running (e.g., 'docker run -p 6379:6379 redis:8.4')", opts.Addr, err)
	}
	return nil
}

// Close terminates the connection to Redis.
func (r *RedisDB) Close() error {
	if r.client == nil {
		return nil
	}
	return r.client.Close()
}

// float32Bytes encodes a vector as little-endian FLOAT32, the layout Redis vector fields expect.
func float32Bytes(v []float64) []byte {
	b := make([]byte, 4*len(v))
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[4*i:], math.Float32bits(float32(f)))
	}
	return b
}

// redisValue converts a Record field into a hash value.
// Metadata maps become JSON, as in MilvusDB.appendToColumn.
func redisValue(v interface{}) (interface{}, error) {
	switch x := v.(type) {
	case Vector:
		return float32Bytes(x), nil
	case []float64:
		return float32Bytes(x), nil
	case []float32:
		f := make([]float64, len(x))
		for i, val := range x {
			f[i] = float64(val)
		}
		return float32Bytes(f), nil
	case map[string]interface{}:
		b, err := json.Marshal(x)
		return string(b), err
	case string, int64, int, float64:
		return x, nil
	default:
		return nil, fmt.Errorf("unsupported field type %T", v)
	}
}

// redisMetric maps raggo metric names to Redis DISTANCE_METRIC values.
func redisMetric(metric string) string {
	switch strings.ToUpper(metric) {
	case "IP":
		return "IP"
	case "COSINE":
		return "COSINE"
	default:
		return "L2" // same fallback as MilvusDB.convertMetricType
	}
}

// distToScore turns a Redis vector distance into a higher-is-better score.
func distToScore(dist float64, metric string) float64 {
	if redisMetric(metric) == "L2" {
		return 1 / (1 + dist)
	}
	return 1 - dist // COSINE and IP distances are 1 - similarity
}

// textQuery turns free text into an OR of its words on the Text field, so
// query-syntax characters in user input can't break FT.HYBRID. No words matches all.
// ponytail: text field fixed to "Text" (every raggo schema uses it); make it a searchParam if that changes.
func textQuery(text string) string {
	words := strings.FieldsFunc(text, func(c rune) bool { return !unicode.IsLetter(c) && !unicode.IsDigit(c) })
	if len(words) == 0 {
		return "*"
	}
	return "@Text:(" + strings.Join(words, "|") + ")"
}

// fuseRRF merges ranked lists by reciprocal rank, normalized so a document
// ranked first in every list scores 1.
func fuseRRF(lists [][]SearchResult, topK int) []SearchResult {
	scores := make(map[int64]float64)
	docs := make(map[int64]SearchResult)
	for _, list := range lists {
		for rank, res := range list {
			scores[res.ID] += 1 / (rrfK + float64(rank+1))
			docs[res.ID] = res
		}
	}
	out := make([]SearchResult, 0, len(docs))
	for id, res := range docs {
		res.Score = scores[id] * (rrfK + 1) / float64(len(lists))
		out = append(out, res)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if len(out) > topK {
		out = out[:topK]
	}
	return out
}

// namePattern allowlists index and field names, which go into query strings
// unparameterized (Redis PARAMS can't bind identifiers).
var namePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

func validName(kind, name string) error {
	if !namePattern.MatchString(name) {
		return fmt.Errorf("invalid %s name %q: use letters, digits and _", kind, name)
	}
	return nil
}

// keyID extracts the numeric ID from a "<collection>:<id>" key.
func keyID(key string) int64 {
	id, _ := strconv.ParseInt(key[strings.LastIndexByte(key, ':')+1:], 10, 64)
	return id
}

// efRuntime reads the HNSW search-time ef from searchParams, as MilvusDB does.
func efRuntime(params map[string]interface{}) int {
	if ef, ok := params["ef"].(int); ok && ef > 0 {
		return ef
	}
	return 10 // Redis default EF_RUNTIME
}
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `go test ./rag -run 'TestTextQuery|TestValidName|TestFuseRRF|TestRedisConnectErrorHidesPassword' -v`
Expected: PASS. `go vet ./rag` may report `RedisDB` as incomplete, which is fine because nothing uses it as a `VectorDB` yet.

- [ ] **Step 6: Commit**

```bash
mkdir -p docs/superpowers/plans && cp /home/azilber/.claude/plans/buzzing-brewing-valiant.md docs/superpowers/plans/2026-09-23-redis-vector-backend.md
git add go.mod go.sum rag/redis.go rag/redis_test.go docs/superpowers/plans/2026-09-23-redis-vector-backend.md
git commit -m "feat(rag): add redis client dependency and vector helpers"
```

---

### Task 2: RedisDB collections, insert, KNN search, factory wiring

**Files:**
- Modify: `rag/redis.go`, appending the methods below.
- Modify: `rag/vector_interface.go:156-167`, the `NewVectorDB` switch.
- Modify: `vectordb.go:41-58`, the `WithType` and `WithAddress` doc comments.
- Test: `rag/redis_test.go`

**Interfaces:**
- Consumes: the helpers from Task 1.
- Produces: `*RedisDB` satisfies `rag.VectorDB`, and `rag.NewVectorDB(&Config{Type: "redis"})` returns it. The test helper `func redisTestDB(t *testing.T) VectorDB` is reused by Task 3.

- [ ] **Step 1: Write the failing integration test** (append to `rag/redis_test.go` and add `"errors"`, `"fmt"`, `"os"` and `"sync"` to the imports)

```go
// redisTestDB connects to REDIS_ADDR (Redis 8.4+) or skips the test.
func redisTestDB(t *testing.T) VectorDB {
	t.Helper()
	addr := os.Getenv("REDIS_ADDR")
	if addr == "" {
		t.Skip("set REDIS_ADDR to a Redis 8.4+ server to run")
	}
	db, err := NewVectorDB(&Config{Type: "redis", Address: addr})
	if err != nil {
		t.Fatal(err)
	}
	if err := db.Connect(context.Background()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

// setupRedisDocs creates a 4-dim COSINE collection with three documents.
func setupRedisDocs(t *testing.T, db VectorDB, col string) {
	t.Helper()
	ctx := context.Background()
	if ok, _ := db.HasCollection(ctx, col); ok {
		must(t, db.DropCollection(ctx, col))
	}
	must(t, db.CreateCollection(ctx, col, Schema{Name: col, Fields: []Field{
		{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
		{Name: "Embedding", DataType: "float_vector", Dimension: 4},
		{Name: "Text", DataType: "varchar", MaxLength: 65535},
		{Name: "Metadata", DataType: "varchar", MaxLength: 65535},
	}}))
	must(t, db.CreateIndex(ctx, col, "Embedding", Index{Type: "HNSW", Metric: "COSINE",
		Parameters: map[string]interface{}{"M": 16, "efConstruction": 200}}))
	must(t, db.Insert(ctx, col, []Record{
		{Fields: map[string]interface{}{"Embedding": []float64{1, 0, 0, 0}, "Text": "golang channels and goroutines", "Metadata": map[string]interface{}{"source": "a.txt"}}},
		{Fields: map[string]interface{}{"Embedding": []float64{0, 1, 0, 0}, "Text": "redis vector search engine", "Metadata": map[string]interface{}{"source": "b.txt"}}},
		{Fields: map[string]interface{}{"Embedding": []float64{0, 0, 1, 0}, "Text": "baking sourdough bread", "Metadata": map[string]interface{}{"source": "c.txt"}}},
	}))
	db.SetColumnNames([]string{"Text", "Metadata"})
}

func TestRedisSearch(t *testing.T) {
	db := redisTestDB(t)
	ctx := context.Background()
	const col = "raggo_test_search"
	setupRedisDocs(t, db, col)

	// Review focus 2: a wrong-length vector must fail loudly, not vanish from the index.
	err := db.Insert(ctx, col, []Record{{Fields: map[string]interface{}{"Embedding": []float64{1, 0}, "Text": "short"}}})
	if err == nil || !strings.Contains(err.Error(), "dimensions") {
		t.Errorf("Insert with 2-dim vector: err = %v, want dimension error", err)
	}

	q := map[string]Vector{"Embedding": {1, 0.1, 0, 0}}
	res, err := db.Search(ctx, col, q, 2, "COSINE", map[string]interface{}{"type": "HNSW", "ef": 64})
	must(t, err)
	if len(res) != 2 || res[0].Fields["Text"] != "golang channels and goroutines" {
		t.Fatalf("Search = %+v, want golang doc first", res)
	}
	if res[0].Score < 0.9 || res[0].Score > 1.0001 || res[0].ID == 0 {
		t.Errorf("top result score=%v id=%v, want ~1 and nonzero id", res[0].Score, res[0].ID)
	}
	// Metadata is stored but not indexed; it must still come back as JSON.
	if md, _ := res[0].Fields["Metadata"].(string); !strings.Contains(md, "a.txt") {
		t.Errorf("Metadata = %q, want JSON containing a.txt", md)
	}

	// Identifiers go into the query string unparameterized, so they are allowlisted.
	if _, err := db.Search(ctx, col, map[string]Vector{"x]=>[KNN": {1, 0, 0, 0}}, 1, "COSINE", nil); err == nil {
		t.Error("Search accepted an invalid field name")
	}

	// A batch larger than insertBatch spans several MULTI/EXECs and all of it lands.
	big := make([]Record, insertBatch+1)
	for i := range big {
		big[i] = Record{Fields: map[string]interface{}{"Embedding": []float64{0, 1, 1, 0}, "Text": "bulk"}}
	}
	must(t, db.Insert(ctx, col, big))
	res, err = db.Search(ctx, col, map[string]Vector{"Embedding": {0, 1, 1, 0}}, insertBatch+1, "COSINE", map[string]interface{}{"ef": 1000})
	must(t, err)
	if len(res) != insertBatch+1 {
		t.Errorf("after bulk insert got %d nearest, want %d", len(res), insertBatch+1)
	}

	// Review focus 7: concurrent Insert/CreateCollection/SetColumnNames (concurrentloader pattern) must not race. Run with -race.
	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_ = db.CreateCollection(ctx, fmt.Sprintf("raggo_test_unused_%d", i), Schema{})
			db.SetColumnNames([]string{"Text", "Metadata"})
			if err := db.Insert(ctx, col, []Record{{Fields: map[string]interface{}{"Embedding": []float64{0, 0, 1, 1}, "Text": "concurrent"}}}); err != nil { // not {0,0,0,1}: that would tie with "fourth doc" below
				t.Error(err)
			}
		}(i)
	}
	wg.Wait()

	// Review focus 6: a cancelled caller context stops the call instead of running to completion.
	cctx, cancel := context.WithCancel(ctx)
	cancel()
	if _, err := db.Search(cctx, col, q, 2, "COSINE", nil); !errors.Is(err, context.Canceled) {
		t.Errorf("Search with cancelled ctx: err = %v, want context.Canceled", err)
	}

	// Review focus 5: a fresh instance (as after a restart) works against the existing index.
	db2 := redisTestDB(t)
	must(t, db2.CreateIndex(ctx, col, "Embedding", Index{Type: "HNSW", Metric: "COSINE"}))
	must(t, db2.Insert(ctx, col, []Record{{Fields: map[string]interface{}{"Embedding": []float64{0, 0, 0, 1}, "Text": "fourth doc"}}}))
	db2.SetColumnNames([]string{"Text"})
	res, err = db2.Search(ctx, col, map[string]Vector{"Embedding": {0, 0, 0, 1}}, 1, "COSINE", nil)
	must(t, err)
	if len(res) != 1 || res[0].Fields["Text"] != "fourth doc" {
		t.Errorf("second instance Search = %+v, want fourth doc", res)
	}

	must(t, db.DropCollection(ctx, col))
	if ok, err := db.HasCollection(ctx, col); ok || err != nil {
		t.Errorf("after drop HasCollection = %v, %v; want false, nil", ok, err)
	}
}
```

- [ ] **Step 2: Start Redis and confirm the test fails**

Run:
```bash
docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4
REDIS_ADDR=localhost:6379 go test ./rag -run TestRedisSearch -v
```
Expected: FAIL to compile, because `*RedisDB` does not implement `VectorDB` (it is missing `HasCollection`).

- [ ] **Step 3: Implement the methods** (append to `rag/redis.go`, and add `"slices"` to its imports)

```go
// idCounterKey holds the AutoID counter. It's a string key, so the hash-only index ignores it.
func idCounterKey(collection string) string { return "raggo:id:" + collection }

// HasCollection reports whether the collection's FT index exists.
func (r *RedisDB) HasCollection(ctx context.Context, name string) (bool, error) {
	err := r.client.Do(ctx, "FT.INFO", name).Err()
	if err == nil {
		return true, nil
	}
	if msg := strings.ToLower(err.Error()); strings.Contains(msg, "unknown index") || strings.Contains(msg, "no such index") {
		return false, nil
	}
	return false, err
}

// DropCollection removes the index, every document hash, and the ID counter.
func (r *RedisDB) DropCollection(ctx context.Context, name string) error {
	r.mu.Lock()
	delete(r.schemas, name)
	r.mu.Unlock()
	if err := r.client.FTDropIndexWithArgs(ctx, name, &redis.FTDropIndexOptions{DeleteDocs: true}).Err(); err != nil {
		return err
	}
	return r.client.Del(ctx, idCounterKey(name)).Err()
}

// CreateCollection records the schema; the FT index is built by CreateIndex,
// because Redis takes the vector metric and HNSW parameters in the same command.
func (r *RedisDB) CreateCollection(ctx context.Context, name string, schema Schema) error {
	schema.Fields = slices.Clone(schema.Fields) // don't alias the caller's slice
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.schemas == nil { // zero-value RedisDB stays usable
		r.schemas = make(map[string]Schema)
	}
	r.schemas[name] = schema
	return nil
}

// CreateIndex issues FT.CREATE over "<collection>:" hashes. It is a no-op when
// the index already exists (a second vector field, or a restarted process).
// ponytail: every vector field shares this call's metric and HNSW params; per-field metrics if a schema ever mixes them.
func (r *RedisDB) CreateIndex(ctx context.Context, collectionName, field string, index Index) error {
	if err := validName("collection", collectionName); err != nil {
		return err
	}
	exists, err := r.HasCollection(ctx, collectionName)
	if err != nil || exists {
		return err
	}
	schema, ok := r.schema(collectionName)
	if !ok {
		return fmt.Errorf("collection %s has no schema: call CreateCollection first", collectionName)
	}
	if index.Type != "HNSW" {
		return fmt.Errorf("unsupported index type: %s", index.Type)
	}
	m, _ := index.Parameters["M"].(int)
	efc, _ := index.Parameters["efConstruction"].(int)

	var fields []*redis.FieldSchema
	for _, f := range schema.Fields {
		switch f.DataType {
		case "float_vector":
			if err := validName("field", f.Name); err != nil {
				return err
			}
			fields = append(fields, &redis.FieldSchema{
				FieldName: f.Name,
				FieldType: redis.SearchFieldTypeVector,
				VectorArgs: &redis.FTVectorArgs{HNSWOptions: &redis.FTHNSWOptions{
					Type:                   "FLOAT32",
					Dim:                    f.Dimension,
					DistanceMetric:         redisMetric(index.Metric),
					MaxEdgesPerNode:        m,   // M
					MaxAllowedEdgesPerNode: efc, // EF_CONSTRUCTION
				}},
			})
		case "varchar":
			if f.Name == "Text" { // the only field textQuery searches
				fields = append(fields, &redis.FieldSchema{FieldName: f.Name, FieldType: redis.SearchFieldTypeText})
			}
		}
		// Other varchars (Metadata JSON) and the int64 AutoID key are stored in the hash
		// and returned by RETURN/LOAD, but not indexed: nothing queries them.
	}
	GlobalLogger.Debug("Creating Redis index", "name", collectionName, "fields", len(fields))
	if err := r.client.FTCreate(ctx, collectionName,
		&redis.FTCreateOptions{OnHash: true, Prefix: []interface{}{collectionName + ":"}},
		fields...).Err(); err != nil {
		return fmt.Errorf("redis FT.CREATE %s: %w", collectionName, err)
	}
	return nil
}

// insertBatch caps records per MULTI/EXEC: Redis is single-threaded, and one EXEC
// with thousands of 6KB vectors would stall every other client.
const insertBatch = 500

// Insert writes each record as a hash at "<collection>:<id>", reserving IDs
// with one INCRBY (AutoID parity with Milvus) and writing HSETs in atomic batches.
func (r *RedisDB) Insert(ctx context.Context, collectionName string, data []Record) error {
	if len(data) == 0 {
		return nil
	}
	// Redis silently skips indexing a hash whose vector has the wrong size, so check it here.
	// ponytail: dims known only for schemas this process created; read FT.INFO if restarts must be checked too.
	dims := make(map[string]int)
	schema, _ := r.schema(collectionName)
	for _, f := range schema.Fields {
		if f.DataType == "float_vector" {
			dims[f.Name] = f.Dimension
		}
	}

	// Convert and validate everything before writing anything.
	hashes := make([][]interface{}, len(data))
	for i, rec := range data {
		values := make([]interface{}, 0, 2*len(rec.Fields))
		for name, v := range rec.Fields {
			val, err := redisValue(v)
			if err != nil {
				return fmt.Errorf("record %d field %s: %w", i, name, err)
			}
			if want, ok := dims[name]; ok {
				if b, _ := val.([]byte); len(b) != 4*want {
					return fmt.Errorf("record %d field %s: vector has %d dimensions, index expects %d", i, name, len(b)/4, want)
				}
			}
			values = append(values, name, val)
		}
		hashes[i] = values
	}

	last, err := r.client.IncrBy(ctx, idCounterKey(collectionName), int64(len(data))).Result()
	if err != nil {
		return fmt.Errorf("redis reserve ids for %s: %w", collectionName, err)
	}
	first := last - int64(len(data)) + 1

	for start := 0; start < len(hashes); start += insertBatch {
		end := min(start+insertBatch, len(hashes))
		pipe := r.client.TxPipeline() // MULTI/EXEC: each batch lands whole or not at all
		for i := start; i < end; i++ {
			pipe.HSet(ctx, fmt.Sprintf("%s:%d", collectionName, first+int64(i)), hashes[i]...)
		}
		if _, err := pipe.Exec(ctx); err != nil {
			GlobalLogger.Error("Failed to insert data", "collection", collectionName, "error", err)
			return fmt.Errorf("redis insert into %s (records %d-%d; earlier batches were written): %w", collectionName, start, end-1, err)
		}
	}
	return nil
}

// Flush is a no-op: Redis indexes writes synchronously.
func (r *RedisDB) Flush(ctx context.Context, collectionName string) error { return nil }

// LoadCollection is a no-op: Redis keeps indexes in memory.
func (r *RedisDB) LoadCollection(ctx context.Context, name string) error { return nil }

// SetColumnNames sets the list of fields to return in search results.
func (r *RedisDB) SetColumnNames(names []string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.columnNames = slices.Clone(names) // caller may reuse its slice
}

// Search runs a KNN query on the single vector field in vectors.
func (r *RedisDB) Search(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}) ([]SearchResult, error) {
	if len(vectors) != 1 {
		return nil, fmt.Errorf("redis search takes exactly one vector field, got %d", len(vectors))
	}
	if topK <= 0 {
		return nil, fmt.Errorf("topK must be positive, got %d", topK)
	}
	var field string
	var vec Vector
	for f, v := range vectors {
		field, vec = f, v
	}
	if err := validName("field", field); err != nil {
		return nil, err
	}

	cols := r.columns()
	returns := []redis.FTSearchReturn{{FieldName: "__dist"}}
	for _, c := range cols {
		returns = append(returns, redis.FTSearchReturn{FieldName: c})
	}
	res, err := r.client.FTSearchWithArgs(ctx, collectionName,
		fmt.Sprintf("*=>[KNN $K @%s $vec EF_RUNTIME $EF AS __dist]", field),
		&redis.FTSearchOptions{
			Params:         map[string]interface{}{"K": topK, "EF": efRuntime(searchParams), "vec": float32Bytes(vec)},
			DialectVersion: 2,
			Return:         returns,
			SortBy:         []redis.FTSearchSortBy{{FieldName: "__dist", Asc: true}},
			Limit:          topK,
		}).Result()
	if err != nil {
		return nil, fmt.Errorf("redis FT.SEARCH %s: %w", collectionName, err)
	}

	results := make([]SearchResult, 0, len(res.Docs))
	for _, doc := range res.Docs {
		dist, err := strconv.ParseFloat(doc.Fields["__dist"], 64)
		if err != nil {
			return nil, fmt.Errorf("doc %s: bad distance %q: %w", doc.ID, doc.Fields["__dist"], err)
		}
		fields := make(map[string]interface{}, len(cols))
		for _, c := range cols {
			if v, ok := doc.Fields[c]; ok {
				fields[c] = v
			}
		}
		results = append(results, SearchResult{ID: keyID(doc.ID), Score: distToScore(dist, metricType), Fields: fields})
	}
	return results, nil
}
```

`HybridSearch` still has to exist for the interface. Add a temporary version now; Task 3 replaces it:

```go
// HybridSearch is implemented in Task 3.
func (r *RedisDB) HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error) {
	return nil, fmt.Errorf("redis hybrid search not implemented")
}
```

In `rag/vector_interface.go`, inside the `NewVectorDB` switch after `case "chromem":`:

```go
	case "redis":
		return newRedisDB(cfg)
```

In `vectordb.go`, add these lines to the doc comments:
- In the `WithType` list: `// - "redis": Redis 8.4+ Query Engine (vector + FT.HYBRID)`
- In the `WithAddress` examples: `// - Redis: "localhost:6379" or "redis://user:pass@host:6379/0"`

- [ ] **Step 4: Run the test and confirm it passes**

Run: `REDIS_ADDR=localhost:6379 go test -race ./rag -run 'TestRedis|TestTextQuery|TestFuseRRF' -v`
Expected: PASS with no race reports. Then run `go build . ./rag/... ./config/... && go vet . ./rag/... ./config/...`, which should be clean.

- [ ] **Step 5: Commit**

```bash
git add rag/redis.go rag/redis_test.go rag/vector_interface.go vectordb.go
git commit -m "feat(rag): add redis vector store with KNN search"
```

---

### Task 3: HybridSearch via FT.HYBRID (RRF, LINEAR, multi-vector)

**Files:**
- Modify: `rag/redis.go`, replacing the temporary `HybridSearch` and adding `hybridArgs`, `toMap` and `parseHybrid`.
- Test: `rag/redis_test.go`

**Interfaces:**
- Consumes: `textQuery`, `fuseRRF`, `efRuntime`, `float32Bytes`, `keyID`, `rrfK` (Task 1); `redisTestDB`, `setupRedisDocs`, `must` (Task 2).
- Produces the `searchParams` keys that `HybridSearch` reads. Task 4 relies on `query_text`.
  - `"query_text"` (string): the user's raw query.
  - `"combine"`: `"RRF"` (the default) or `"LINEAR"`.
  - `"alpha"` and `"beta"` (float64): LINEAR weights, default 0.5 each.
  - `"ef"` (int)

- [ ] **Step 1: Write the failing tests** (append to `rag/redis_test.go`)

```go
func TestRedisHybridSearch(t *testing.T) {
	db := redisTestDB(t)
	ctx := context.Background()
	const col = "raggo_test_hybrid"
	setupRedisDocs(t, db, col)
	defer db.DropCollection(ctx, col)

	q := map[string]Vector{"Embedding": {1, 0.1, 0, 0}} // nearest: golang doc
	for _, combine := range []string{"RRF", "LINEAR"} {
		// Review focus 3: "?" must not reach Redis query syntax.
		res, err := db.HybridSearch(ctx, col, q, 3, "COSINE",
			map[string]interface{}{"query_text": "sourdough bread?", "combine": combine, "ef": 64}, nil)
		must(t, err)
		texts := map[interface{}]bool{}
		for _, r := range res {
			texts[r.Fields["Text"]] = true
		}
		if !texts["golang channels and goroutines"] || !texts["baking sourdough bread"] {
			t.Errorf("%s: results %+v, want both the vector hit and the text hit", combine, res)
		}
		// Review focus 4: RRF scores are normalized into (0,1].
		if combine == "RRF" && (res[0].Score <= 0 || res[0].Score > 1+1e-9) { // epsilon: float sum of 1/(k+r)
			t.Errorf("RRF top score %v not in (0,1]", res[0].Score)
		}
	}

	// No query text: the vector ranking alone decides.
	res, err := db.HybridSearch(ctx, col, q, 1, "COSINE", nil, nil)
	must(t, err)
	if len(res) != 1 || res[0].Fields["Text"] != "golang channels and goroutines" {
		t.Errorf("text-less hybrid = %+v, want golang doc", res)
	}
}

func TestRedisHybridMultiVector(t *testing.T) {
	db := redisTestDB(t)
	ctx := context.Background()
	const col = "raggo_test_multivec"
	if ok, _ := db.HasCollection(ctx, col); ok {
		must(t, db.DropCollection(ctx, col))
	}
	defer db.DropCollection(ctx, col)
	must(t, db.CreateCollection(ctx, col, Schema{Name: col, Fields: []Field{
		{Name: "ID", DataType: "int64", PrimaryKey: true, AutoID: true},
		{Name: "A", DataType: "float_vector", Dimension: 2},
		{Name: "B", DataType: "float_vector", Dimension: 2},
		{Name: "Text", DataType: "varchar", MaxLength: 1024},
	}}))
	idx := Index{Type: "HNSW", Metric: "COSINE"}
	must(t, db.CreateIndex(ctx, col, "A", idx))
	must(t, db.CreateIndex(ctx, col, "B", idx)) // no-op: index already covers B
	must(t, db.Insert(ctx, col, []Record{
		{Fields: map[string]interface{}{"A": []float64{1, 0}, "B": []float64{1, 0}, "Text": "both match"}},
		{Fields: map[string]interface{}{"A": []float64{1, 0}, "B": []float64{0, 1}, "Text": "only A matches"}},
		{Fields: map[string]interface{}{"A": []float64{0, 1}, "B": []float64{0, 1}, "Text": "neither matches"}},
	}))
	db.SetColumnNames([]string{"Text"})

	res, err := db.HybridSearch(ctx, col, map[string]Vector{"A": {1, 0}, "B": {1, 0}}, 2, "COSINE", nil, nil)
	must(t, err)
	if len(res) != 2 || res[0].Fields["Text"] != "both match" {
		t.Errorf("multi-vector hybrid = %+v, want 'both match' first, 2 results", res)
	}
}
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `REDIS_ADDR=localhost:6379 go test ./rag -run TestRedisHybrid -v`
Expected: FAIL with `redis hybrid search not implemented`.

- [ ] **Step 3: Write the implementation** (replace the temporary `HybridSearch` in `rag/redis.go`)

```go
// HybridSearch fuses BM25 text search on Text with vector KNN using FT.HYBRID.
// The query text comes from searchParams["query_text"]; without it only the vector
// ranking counts. Fusion: searchParams["combine"] = "RRF" (default) or "LINEAR"
// with optional "alpha"/"beta" weights. FT.HYBRID takes one vector field, so
// several fields run one FT.HYBRID each (pipelined) and are merged with RRF.
func (r *RedisDB) HybridSearch(ctx context.Context, collectionName string, vectors map[string]Vector, topK int, metricType string, searchParams map[string]interface{}, reranker interface{}) ([]SearchResult, error) {
	if reranker != nil {
		return nil, fmt.Errorf("redis fuses inside FT.HYBRID; set searchParams[\"combine\"] instead of passing a reranker")
	}
	if len(vectors) == 0 {
		return nil, fmt.Errorf("hybrid search needs at least one vector")
	}
	if topK <= 0 {
		return nil, fmt.Errorf("topK must be positive, got %d", topK)
	}
	text, _ := searchParams["query_text"].(string)
	combine, _ := searchParams["combine"].(string)
	linear := strings.EqualFold(combine, "LINEAR")
	cols := r.columns()

	pipe := r.client.Pipeline()
	cmds := make([]*redis.Cmd, 0, len(vectors))
	for field, vec := range vectors {
		if err := validName("field", field); err != nil {
			return nil, err
		}
		cmds = append(cmds, pipe.Do(ctx, hybridArgs(collectionName, textQuery(text), field, vec, topK, linear, searchParams, cols)...))
	}
	if _, err := pipe.Exec(ctx); err != nil {
		return nil, fmt.Errorf("redis FT.HYBRID %s: %w", collectionName, err)
	}

	lists := make([][]SearchResult, len(cmds))
	for i, cmd := range cmds {
		list, err := parseHybrid(cmd.Val(), cols)
		if err != nil {
			return nil, err
		}
		if !linear { // FT.HYBRID RRF fuses 2 lists (text, vector): max 2/(k+1). Scale into [0,1].
			for j := range list {
				list[j].Score *= (rrfK + 1) / 2
			}
		}
		lists[i] = list
	}
	if len(lists) == 1 {
		return lists[0], nil
	}
	return fuseRRF(lists, topK), nil
}

// hybridArgs builds one FT.HYBRID command (syntax: redis.io/docs/latest/commands/ft.hybrid).
func hybridArgs(index, query, field string, vec Vector, topK int, linear bool, params map[string]interface{}, cols []string) []interface{} {
	window := max(topK, 20) // 20 is Redis's default fusion window
	args := []interface{}{"FT.HYBRID", index,
		"SEARCH", query,
		"VSIM", "@" + field, "$vec", "KNN", 4, "K", topK, "EF_RUNTIME", efRuntime(params)}
	if linear {
		alpha, ok := params["alpha"].(float64)
		if !ok {
			alpha = 0.5
		}
		beta, ok := params["beta"].(float64)
		if !ok {
			beta = 0.5
		}
		args = append(args, "COMBINE", "LINEAR", 6, "ALPHA", alpha, "BETA", beta, "WINDOW", window)
	} else {
		args = append(args, "COMBINE", "RRF", 4, "CONSTANT", rrfK, "WINDOW", window)
	}
	args = append(args, "LIMIT", 0, topK, "LOAD", len(cols)+2, "@__key", "@__score")
	for _, c := range cols {
		args = append(args, "@"+c)
	}
	return append(args, "PARAMS", 2, "vec", float32Bytes(vec))
}

// toMap reads a RESP2 flat key/value array or a RESP3 map.
func toMap(v interface{}) map[string]interface{} {
	out := make(map[string]interface{})
	switch x := v.(type) {
	case []interface{}:
		for i := 0; i+1 < len(x); i += 2 {
			out[fmt.Sprint(x[i])] = x[i+1]
		}
	case map[interface{}]interface{}:
		for k, val := range x {
			out[fmt.Sprint(k)] = val
		}
	}
	return out
}

// parseHybrid converts an FT.HYBRID reply ({total_results, results: [{__key, __score, fields...}], ...}).
// A row missing __key or __score is an error, not a silent ID 0 / score 0.
func parseHybrid(reply interface{}, cols []string) ([]SearchResult, error) {
	rows, ok := toMap(reply)["results"].([]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected FT.HYBRID reply: %v", reply)
	}
	results := make([]SearchResult, 0, len(rows))
	for _, row := range rows {
		m := toMap(row)
		key, hasKey := m["__key"]
		rawScore, hasScore := m["__score"]
		if !hasKey || !hasScore {
			return nil, fmt.Errorf("FT.HYBRID row missing __key/__score: %v", m)
		}
		score, err := strconv.ParseFloat(fmt.Sprint(rawScore), 64)
		if err != nil {
			return nil, fmt.Errorf("FT.HYBRID bad score %v: %w", rawScore, err)
		}
		fields := make(map[string]interface{}, len(cols))
		for _, c := range cols {
			if v, ok := m[c]; ok {
				fields[c] = v
			}
		}
		results = append(results, SearchResult{ID: keyID(fmt.Sprint(key)), Score: score, Fields: fields})
	}
	return results, nil
}
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `REDIS_ADDR=localhost:6379 go test ./rag -run 'TestRedis|TestTextQuery|TestFuseRRF' -v`
Expected: PASS.

If `parseHybrid` returns `unexpected FT.HYBRID reply` or `row missing __key/__score`, then the reply shape or key names differ from the docs. The error message prints the row it received. In that case:
1. Temporarily add `t.Logf("%#v", cmd.Val())` to see the real shape.
2. Fix the key names in `parseHybrid`.
3. Remove the log.

Don't loosen the test asserts.

- [ ] **Step 5: Commit**

```bash
git add rag/redis.go rag/redis_test.go
git commit -m "feat(rag): redis hybrid search via FT.HYBRID with RRF/LINEAR and multi-vector fusion"
```

---

### Task 4: Pass query text from the raggo callers

**Files:**
- Modify: `retriever.go`, adding the `withQueryText` helper and using it in the `Retrieve` hybrid branch (~line 136).
- Modify: `rag.go`, in `hybridSearch` (~line 799).
- Test: `retriever_test.go` (new, root package).

**Interfaces:**
- Consumes: the `searchParams["query_text"]` contract from Task 3.
- Produces: `func withQueryText(params map[string]interface{}, query string) map[string]interface{}` (unexported, package `raggo`).

- [ ] **Step 1: Write the failing test**

`retriever_test.go`:

```go
package raggo

import "testing"

func TestWithQueryText(t *testing.T) {
	base := map[string]interface{}{"ef": 64}
	got := withQueryText(base, "hello")
	if got["query_text"] != "hello" || got["ef"] != 64 {
		t.Errorf("withQueryText = %v", got)
	}
	if _, leaked := base["query_text"]; leaked {
		t.Error("withQueryText mutated the shared config map")
	}
	if withQueryText(nil, "x")["query_text"] != "x" {
		t.Error("nil params must still work")
	}
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `go test . -run TestWithQueryText -v`
Expected: FAIL with `undefined: withQueryText`.

- [ ] **Step 3: Implement it**

Append to `retriever.go`:

```go
// withQueryText copies params and adds the raw query for backends whose hybrid
// search needs the text (Redis FT.HYBRID); other backends ignore the key.
func withQueryText(params map[string]interface{}, query string) map[string]interface{} {
	out := make(map[string]interface{}, len(params)+1)
	for k, v := range params {
		out[k] = v
	}
	out["query_text"] = query
	return out
}
```

In `retriever.go` `Retrieve`, hybrid branch, change the `r.config.SearchParams,` argument of `r.vectorDB.HybridSearch(...)` to:

```go
			withQueryText(r.config.SearchParams, query),
```

In `rag.go` `hybridSearch`, change the argument `r.config.SearchParams, // Use the config's search params` to:

```go
		withQueryText(r.config.SearchParams, query),
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `go test . -run TestWithQueryText -v && go build . ./rag/... ./config/... && go vet . ./rag/... ./config/...`
Expected: PASS, and the build and vet are clean.

- [ ] **Step 5: Commit**

```bash
git add retriever.go retriever_test.go rag.go
git commit -m "feat: pass query text to hybrid search for text-aware backends"
```

---

### Task 5: RAG options and the ContextualRAG Redis option

**Files:**
- Modify: `rag.go`, adding `SetDBType` after `SetDBAddress` (~line 229) and `WithRedis` after `WithMilvus` (~line 345).
- Modify: `contextual_rag.go`:
  - `ContextualRAGConfig` (~line 62)
  - `DefaultContextualConfig` (~line 106)
  - the merge block in `NewContextualRAG` (~line 138)
  - `initializeRAG` (~lines 206, 253, 264)
- Modify: `CLAUDE.md`, the vector store section.
- Test: `contextual_rag_test.go` (new).

**Interfaces:**
- Consumes: the `"redis"` type from Task 2.
- Produces:
  - `func SetDBType(dbType string) RAGOption`
  - `func WithRedis(collection string) RAGOption`
  - `ContextualRAGConfig.DBType string` and `ContextualRAGConfig.DBAddress string`

- [ ] **Step 1: Write the failing test**

`contextual_rag_test.go`:

```go
package raggo

import "testing"

func TestRedisOptions(t *testing.T) {
	c := DefaultRAGConfig()
	WithRedis("docs")(c)
	if c.DBType != "redis" || c.DBAddress != "localhost:6379" || c.Collection != "docs" {
		t.Errorf("WithRedis -> %s %s %s", c.DBType, c.DBAddress, c.Collection)
	}
	SetDBType("milvus")(c)
	if c.DBType != "milvus" {
		t.Errorf("SetDBType -> %s", c.DBType)
	}
	d := DefaultContextualConfig()
	if d.DBType != "milvus" || d.DBAddress != "localhost:19530" {
		t.Errorf("contextual defaults changed: %s %s", d.DBType, d.DBAddress)
	}
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `go test . -run TestRedisOptions -v`
Expected: FAIL with `undefined: WithRedis`.

- [ ] **Step 3: Implement it**

`rag.go`, after `SetDBAddress`:

```go
// SetDBType selects the vector database backend ("milvus", "redis", "memory", "chromem").
func SetDBType(dbType string) RAGOption {
	return func(c *RAGConfig) {
		c.DBType = dbType
	}
}
```

`rag.go`, after `WithMilvus`:

```go
// WithRedis configures Redis (8.4+) as the vector database with the specified collection.
//
//	rag, err := raggo.NewRAG(
//	    raggo.WithRedis("my_documents"),
//	)
func WithRedis(collection string) RAGOption {
	return func(c *RAGConfig) {
		c.DBType = "redis"
		c.DBAddress = "localhost:6379"
		c.Collection = collection
	}
}
```

`contextual_rag.go`, add to `ContextualRAGConfig` after `MinScore`:

```go
	// DBType selects the vector database ("milvus" or "redis")
	DBType string

	// DBAddress is the vector database address (e.g., "localhost:19530", "localhost:6379")
	DBAddress string
```

In `DefaultContextualConfig`, add:

```go
		DBType:       "milvus",
		DBAddress:    "localhost:19530",
```

In the `NewContextualRAG` merge block, after the `MinScore` default:

```go
		if config.DBType == "" {
			config.DBType = defaultConfig.DBType
		}
		if config.DBAddress == "" {
			config.DBAddress = defaultConfig.DBAddress
		}
```

In `initializeRAG`, replace the three hardcoded Milvus values:

```go
	vectorDB, err := NewVectorDB(
		WithType(config.DBType),
		WithAddress(config.DBAddress),
		WithTimeout(5*time.Minute),
	)
```

```go
	ragOpts := []RAGOption{
		WithOpenAI(config.APIKey),
		SetDBType(config.DBType),
		SetDBAddress(config.DBAddress),
		SetCollection(config.Collection),
	}
```

```go
		WithRetrieveDB(config.DBType, config.DBAddress),
```

In `CLAUDE.md`, "Adding a vector store backend" section: replace the sentence "`ContextualRAG` hardcodes `WithType("milvus")` (`contextual_rag.go`). `SimpleRAG` and `RAG` take the type from their config's `DBType`." with:

```markdown
`ContextualRAG`, `SimpleRAG` and `RAG` all take the backend from their config's `DBType`/`DBAddress` (defaults are Milvus at `localhost:19530`).

Redis (`rag/redis.go`) needs Redis 8.4+ (`FT.HYBRID`). Its hybrid search reads the user's text from `searchParams["query_text"]`, which `withQueryText` (`retriever.go`) adds. Integration tests run with `REDIS_ADDR=localhost:6379 go test ./rag -run TestRedis -v` against `docker run -p 6379:6379 redis:8.4`.
```

- [ ] **Step 4: Run all checks and confirm they pass**

Run:
```bash
go test . -run 'TestRedisOptions|TestWithQueryText' -v
REDIS_ADDR=localhost:6379 go test -race ./rag -v
go build . ./rag/... ./config/... && go vet . ./rag/... ./config/...
```
Expected: everything passes, and the build and vet are clean.

- [ ] **Step 5: Commit and stop Redis**

```bash
git add rag.go contextual_rag.go contextual_rag_test.go CLAUDE.md
git commit -m "feat: selectable vector DB for ContextualRAG, add WithRedis/SetDBType"
docker stop raggo-redis
```

---

## Self-review
- **Spec coverage:**

  | Requirement | Task |
  |---|---|
  | Redis backend + factory | 2 |
  | KNN search | 2 |
  | `FT.HYBRID` hybrid search | 3 |
  | LINEAR option | 3 |
  | Multi-vector hybrid | 3 |
  | `query_text` wiring | 4 |
  | ContextualRAG Redis option | 5 |
  | Docs | 2 (vectordb.go), 5 (CLAUDE.md) |
  | Review Focus 1–5 | pinned in Tasks 1, 2 and 3 as listed |

- **Names used across tasks:** `redisTestDB`, `setupRedisDocs`, `must`, `withQueryText`, `SetDBType`, `WithRedis`, `rrfK`, `fuseRRF`, `textQuery`, `efRuntime`, `float32Bytes`, `keyID`. Each is defined once and used with the same signature everywhere.
- **Known risks:** two `FT.HYBRID` details are taken from the docs and haven't been run against a server yet. Task 3 Step 4 covers finding and fixing them:
  - The reply field names `__key` and `__score`, which come from the docs' reserved-field list.
  - Whether `SEARCH "*"` (the no-text case) is accepted. If it isn't, fall back to plain `Search` when `textQuery` returns `*`.
