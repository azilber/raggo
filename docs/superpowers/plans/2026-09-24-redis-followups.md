# Redis Follow-ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Execution method chosen:** Native (superpowers:executing-plans), with one whole-branch review at the end.
> After approval, copy this file to `docs/superpowers/plans/2026-09-24-redis-followups.md` and commit it with Task 1.

**Goal:** Fix the three minor issues deferred from the Redis backend review, and raise the module's Go version to the system toolchain's 1.27.1.

**Architecture:**
- **`keyID`** now returns an error for a key without a numeric ID suffix, instead of silently returning ID 0.
- **LINEAR weights:** a new `floatParam` helper accepts `alpha`/`beta` as any Go number type and rejects non-numbers. `HybridSearch` reads them before any Redis call and passes them to `hybridArgs`.
- **`go.mod`:** the `go` line becomes `1.27.1`.
- **`CLAUDE.md`:** the stale "no tests" line is replaced with the real test commands.

**Tech Stack:** Go 1.27.1, `github.com/redis/go-redis/v9`, Redis 8.4 (docker) for the integration checks.

**Spec:** this chat request plus the three deferred minors from the review of PR #1:
- `CLAUDE.md:15` still says there are no `_test.go` files.
- `keyID` swallows parse errors, so a bad key becomes ID 0.
- An integer `alpha`/`beta` silently falls back to 0.5.

The user confirmed the version as `go 1.27.1`.

## Global Constraints
- **Branch:** do the work on a new branch, `redis-followups`, created from `main` (HEAD `43d8d0b`). Never commit to `main`.
- **Go version:** `go.mod` gets exactly `go 1.27.1`. Every consumer of raggo will then need Go 1.27.1 or later.
- **Build and vet:** use `go build . ./rag/... ./config/...` and `go vet . ./rag/... ./config/...`. Never use `./...`, because the `examples/` mains clash.
- **Shadowed `max`:** the `rag` package has its own `func max(a, b int) int` in `rag/chunk.go:220`, which shadows the builtin. Don't call `max` or `min` on `int64` or `float64` in package `rag`.
- **Commits:** each commit message ends with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.

## Review Focus
1. **A result key without a numeric suffix** (such as `col:abc` or `nocolon`): Search and HybridSearch must return an error naming the key, not a result with ID 0. Pinned by `TestKeyID` (Task 1).
2. **`alpha`/`beta` that come from JSON config (float64) or are written as literals (int, int64, float32):** all must be honoured. Pinned by `TestFloatParam` (Task 2).
3. **A non-numeric weight** (for example `"alpha": "0.3"`): `HybridSearch` must return an error before sending anything to Redis, rather than silently using 0.5. Pinned by the string-weight check in `TestRedisHybridSearch` (Task 2).
4. **Integer weights reaching Redis:** `alpha: 0, beta: 1` must rank exactly like `0.0, 1.0`, not like the 0.5 default. Pinned by the int-vs-float check in `TestRedisHybridSearch` (Task 2).
5. **Documented commands that don't work:** every command added to `CLAUDE.md` is run in Task 3, Step 3.

---

### Task 1: `keyID` reports bad keys

**Files:**
- Modify: `rag/redis.go:187-191` (`keyID`), plus its two callers: `Search` (~line 453) and `parseHybrid` (~line 636).
- Test: `rag/redis_test.go`. Put `TestKeyID` right after `TestValidName`, which follows the source order.

**Interfaces:**
- Produces: `func keyID(key string) (int64, error)`, unexported, in package `rag`.

- [ ] **Step 0: Create the branch and save the plan**

```bash
git checkout -b redis-followups main
mkdir -p docs/superpowers/plans && cp /home/azilber/.claude/plans/buzzing-brewing-valiant.md docs/superpowers/plans/2026-09-24-redis-followups.md
```

- [ ] **Step 1: Write the failing test** (insert after `TestValidName` in `rag/redis_test.go`)

```go
func TestKeyID(t *testing.T) {
	tests := []struct {
		name    string
		key     string
		want    int64
		wantErr bool
	}{
		{name: "simple key", key: "docs:42", want: 42},
		{name: "last colon wins", key: "a:b:7", want: 7},
		{name: "non-numeric suffix", key: "docs:abc", wantErr: true},
		{name: "empty suffix", key: "docs:", wantErr: true},
		{name: "no colon", key: "nocolon", wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := keyID(tt.key)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), tt.key) {
					t.Errorf("keyID(%q) err = %v, want error naming the key", tt.key, err)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Errorf("keyID(%q) = %d, %v; want %d, nil", tt.key, got, err, tt.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `go test ./rag -run TestKeyID -v`
Expected: FAIL to compile with `assignment mismatch: 2 variables but keyID returns 1 value`.

- [ ] **Step 3: Implement**

Replace `keyID` in `rag/redis.go`:

```go
// keyID extracts the numeric ID from a "<collection>:<id>" key.
func keyID(key string) (int64, error) {
	i := strings.LastIndexByte(key, ':')
	if i < 0 {
		return 0, fmt.Errorf("redis key %q has no <collection>:<id> form", key)
	}
	id, err := strconv.ParseInt(key[i+1:], 10, 64)
	if err != nil {
		return 0, fmt.Errorf("redis key %q has no numeric ID: %w", key, err)
	}
	return id, nil
}
```

In `Search`, replace the line `results = append(results, SearchResult{ID: keyID(doc.ID), Score: distToScore(dist, metricType), Fields: fields})` with:

```go
		id, err := keyID(doc.ID)
		if err != nil {
			return nil, err
		}
		results = append(results, SearchResult{ID: id, Score: distToScore(dist, metricType), Fields: fields})
```

In `parseHybrid`, replace the line `results = append(results, SearchResult{ID: keyID(fmt.Sprint(key)), Score: score, Fields: fields})` with:

```go
		id, err := keyID(fmt.Sprint(key))
		if err != nil {
			return nil, err
		}
		results = append(results, SearchResult{ID: id, Score: score, Fields: fields})
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run:
```bash
docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4
REDIS_ADDR=localhost:6379 go test -race ./rag -v
```
Expected: PASS, with `TestKeyID` and its 5 subtests plus every existing `TestRedis*` test.

- [ ] **Step 5: Commit**

```bash
git add rag/redis.go rag/redis_test.go docs/superpowers/plans/2026-09-24-redis-followups.md
git commit -m "fix(rag): keyID returns an error instead of silent ID 0"
```

---

### Task 2: numeric `alpha`/`beta` of any Go number type

**Files:**
- Modify: `rag/redis.go`:
  - Add `floatParam` next to `efRuntime` (~line 193).
  - In `HybridSearch`, read the weights before any Redis call.
  - Change `hybridArgs` to take `alpha, beta float64` instead of reading them from params (~lines 570-590).
- Test: `rag/redis_test.go`. Put `TestFloatParam` after `TestKeyID`, and add checks inside `TestRedisHybridSearch`.

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `func floatParam(params map[string]interface{}, key string, def float64) (float64, error)`
  - `func hybridArgs(index, query, field string, vec Vector, topK int, linear bool, alpha, beta float64, params map[string]interface{}, cols []string) []interface{}`

- [ ] **Step 1: Write the failing tests**

Insert after `TestKeyID` in `rag/redis_test.go`:

```go
func TestFloatParam(t *testing.T) {
	tests := []struct {
		name    string
		params  map[string]interface{}
		want    float64
		wantErr bool
	}{
		{name: "absent uses default", params: map[string]interface{}{}, want: 0.5},
		{name: "nil map uses default", params: nil, want: 0.5},
		{name: "float64 from JSON", params: map[string]interface{}{"alpha": 0.3}, want: 0.3},
		{name: "float32", params: map[string]interface{}{"alpha": float32(0.25)}, want: 0.25},
		{name: "int literal", params: map[string]interface{}{"alpha": 1}, want: 1},
		{name: "int64", params: map[string]interface{}{"alpha": int64(2)}, want: 2},
		{name: "string is an error", params: map[string]interface{}{"alpha": "0.3"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := floatParam(tt.params, "alpha", 0.5)
			if tt.wantErr {
				if err == nil || !strings.Contains(err.Error(), "alpha") {
					t.Errorf("err = %v, want error naming alpha", err)
				}
				return
			}
			if err != nil || got != tt.want {
				t.Errorf("floatParam = %v, %v; want %v, nil", got, err, tt.want)
			}
		})
	}
}
```

In `TestRedisHybridSearch`, insert these checks just before the comment `// No query text: the vector ranking alone decides.`:

```go
	// Integer weights are honoured exactly like the equal float weights (before the fix, ints fell back to 0.5).
	// The docs don't say which half ALPHA weights, so compare int vs float instead of assuming an order.
	linearTexts := func(alpha, beta interface{}) []interface{} {
		t.Helper()
		res, err := db.HybridSearch(ctx, col, q, 3, "COSINE",
			map[string]interface{}{"query_text": "sourdough bread", "combine": "LINEAR", "alpha": alpha, "beta": beta}, nil)
		must(t, err)
		var out []interface{}
		for _, r := range res {
			out = append(out, r.Fields["Text"])
		}
		return out
	}
	ints, floats, defaults := linearTexts(0, 1), linearTexts(0.0, 1.0), linearTexts(0.5, 0.5)
	if fmt.Sprint(ints) != fmt.Sprint(floats) {
		t.Errorf("LINEAR int weights %v ranked differently from float weights %v", ints, floats)
	}
	if fmt.Sprint(ints) == fmt.Sprint(defaults) {
		t.Logf("note: 0/1 and 0.5/0.5 rank the same here (%v); the int-vs-float check still pins the fix", ints)
	}
	// A non-numeric weight is an error, not a silent 0.5.
	if _, err := db.HybridSearch(ctx, col, q, 3, "COSINE",
		map[string]interface{}{"query_text": "bread", "combine": "LINEAR", "alpha": "0.3"}, nil); err == nil {
		t.Error("string alpha accepted")
	}
```

The existing line after these checks, `res, err := db.HybridSearch(ctx, col, q, 1, "COSINE", nil, nil)`, stays as it is: the new checks declare nothing at function scope.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `REDIS_ADDR=localhost:6379 go test ./rag -run 'TestFloatParam|TestRedisHybridSearch' -v`
Expected: FAIL to compile with `undefined: floatParam`.

- [ ] **Step 3: Implement**

Add after `efRuntime` in `rag/redis.go`:

```go
// floatParam reads a numeric searchParam given as any Go number type. Absent
// means def; any other type is an error rather than a silent fallback.
func floatParam(params map[string]interface{}, key string, def float64) (float64, error) {
	v, ok := params[key]
	if !ok {
		return def, nil
	}
	switch x := v.(type) {
	case float64:
		return x, nil
	case float32:
		return float64(x), nil
	case int:
		return float64(x), nil
	case int32:
		return float64(x), nil
	case int64:
		return float64(x), nil
	default:
		return 0, fmt.Errorf("searchParams[%q] must be a number, got %T", key, v)
	}
}
```

In `HybridSearch`, replace:

```go
	combine, _ := searchParams["combine"].(string)
	linear := strings.EqualFold(combine, "LINEAR")
```

with the version below, and move it up so it runs right after the `validName` loop, before `textQuery` and `textMatches`. That way a bad weight fails before any Redis call:

```go
	combine, _ := searchParams["combine"].(string)
	linear := strings.EqualFold(combine, "LINEAR")
	alpha, err := floatParam(searchParams, "alpha", 0.5)
	if err != nil {
		return nil, err
	}
	beta, err := floatParam(searchParams, "beta", 0.5)
	if err != nil {
		return nil, err
	}
```

The later `matches, err := r.textMatches(...)` line can stay as it is. `err` is now declared earlier, but `:=` is legal because `matches` is new.

In `HybridSearch`'s pipeline loop, change the `hybridArgs` call to:

```go
		cmds = append(cmds, pipe.Do(ctx, hybridArgs(collectionName, query, field, vec, topK, linear, alpha, beta, searchParams, cols)...))
```

Replace the `hybridArgs` signature and its LINEAR branch:

```go
func hybridArgs(index, query, field string, vec Vector, topK int, linear bool, alpha, beta float64, params map[string]interface{}, cols []string) []interface{} {
	window := max(topK, 20) // 20 is Redis's default fusion window
	args := []interface{}{"FT.HYBRID", index,
		"SEARCH", query,
		"VSIM", "@" + field, "$vec", "KNN", 4, "K", topK, "EF_RUNTIME", efRuntime(params)}
	if linear {
		args = append(args, "COMBINE", "LINEAR", 6, "ALPHA", alpha, "BETA", beta, "WINDOW", window)
	} else {
		args = append(args, "COMBINE", "RRF", 4, "CONSTANT", rrfK, "WINDOW", window)
	}
```

The rest of `hybridArgs` (LIMIT, LOAD, PARAMS) is unchanged. Update the `HybridSearch` doc comment line `// with optional "alpha"/"beta" weights.` to `// with optional numeric "alpha"/"beta" weights (default 0.5 each).`

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `REDIS_ADDR=localhost:6379 go test -race ./rag -v`
Expected: PASS, with `TestFloatParam` and its 7 subtests and `TestRedisHybridSearch` including the new checks. Then run `go vet ./rag`, which should be clean.

- [ ] **Step 5: Commit**

```bash
git add rag/redis.go rag/redis_test.go
git commit -m "fix(rag): accept any numeric LINEAR alpha/beta, reject non-numbers"
```

---

### Task 3: Go 1.27.1 and CLAUDE.md test commands

**Files:**
- Modify: `go.mod` (the `go` line), and `go.sum` if `tidy` touches it.
- Modify: `CLAUDE.md`, replacing line 15 and trimming the duplicate test sentence at line 45.

**Interfaces:** none.

- [ ] **Step 1: Raise the Go version**

Run:
```bash
go mod edit -go=1.27.1
go mod tidy
grep -n '^go \|^toolchain' go.mod
```
Expected: `go 1.27.1`, and no `toolchain` line, because the local toolchain is go1.27.1.

- [ ] **Step 2: Update CLAUDE.md**

Replace line 15:

```markdown
- There are no `_test.go` files yet. To run a single test once some exist: `go test ./rag -run TestName -v`.
```

with:

```markdown
- Unit tests: `go test -race . ./rag/...`. Single test: `go test ./rag -run TestName -v`.
- Redis integration tests skip unless `REDIS_ADDR` is set: `docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4`, then `REDIS_ADDR=localhost:6379 go test -race ./rag -run TestRedis -v`.
- End-to-end RAG test (Gemini embeddings + generation on Redis) is behind a build tag: `REDIS_ADDR=localhost:6379 GEMINI_API_KEY=... go test -tags=integration -run TestGeminiRAG -v .`
```

In the Redis paragraph (line 45), delete the trailing sentence `Integration tests run with ``REDIS_ADDR=localhost:6379 go test ./rag -run TestRedis -v`` against ``docker run -p 6379:6379 redis:8.4``.`, because the Commands section now covers it.

- [ ] **Step 3: Verify every documented command, and the whole suite on Go 1.27.1**

Run:
```bash
go build . ./rag/... ./config/... && go vet -tags=integration . ./rag/... ./config/...
go test -race . ./rag/...
go test ./rag -run TestKeyID -v
REDIS_ADDR=localhost:6379 go test -race ./rag -run TestRedis -v
REDIS_ADDR=localhost:6379 GEMINI_API_KEY=<user's key from the session> go test -tags=integration -run TestGeminiRAG -v .
```
Expected: build and vet are clean, and every test command passes. The Gemini run needs the key the user provided. Pass it only through the environment, never write it into a file in the repo, and skip this line if the key isn't available (the test then reports SKIP).

- [ ] **Step 4: Commit, then stop Redis**

```bash
git add go.mod go.sum CLAUDE.md
git commit -m "chore: require Go 1.27.1; document test commands in CLAUDE.md"
docker stop raggo-redis
```

---

## Self-review
- **Spec coverage:**

  | Requirement | Task |
  |---|---|
  | CLAUDE.md stale line | 3 |
  | `keyID` silent 0 | 1 |
  | int `alpha`/`beta` | 2 |
  | Go 1.27.1 | 3 |
  | Review Focus 1–5 | 1, 2, 2, 2, 3 |

- **Placeholders:** none. The Gemini key is supplied at run time on purpose and is never written down.
- **Type consistency:**
  - `keyID` returns `(int64, error)` at both call sites.
  - `floatParam(params, key, def)` returns `(float64, error)`.
  - `hybridArgs` takes the new `alpha, beta float64` in its single caller.
