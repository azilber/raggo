// Package rag provides retrieval-augmented generation capabilities.
package rag

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"slices"
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
	mu          sync.RWMutex // guards columnNames and schemas, as in MemoryDB (callers Insert from goroutines)
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
	opts.Protocol = 2                 // go-redis parses FT.SEARCH replies only on RESP2
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

// floatParam reads a numeric searchParam given as any Go number type or a
// json.Number (from a UseNumber decoder). Absent means def. Anything else, or a
// NaN, infinite or negative value, is an error rather than a silent fallback:
// Redis would accept NaN and return scrambled rankings.
func floatParam(params map[string]interface{}, key string, def float64) (float64, error) {
	v, ok := params[key]
	if !ok {
		return def, nil
	}
	var f float64
	if n, isNum := v.(json.Number); isNum {
		var err error
		if f, err = n.Float64(); err != nil {
			return 0, fmt.Errorf("searchParams[%q]: %w", key, err)
		}
	} else {
		rv := reflect.ValueOf(v)
		switch {
		case rv.CanInt():
			f = float64(rv.Int())
		case rv.CanUint():
			f = float64(rv.Uint())
		case rv.CanFloat():
			f = rv.Float()
		default:
			return 0, fmt.Errorf("searchParams[%q] must be a number, got %T", key, v)
		}
	}
	if math.IsNaN(f) || math.IsInf(f, 0) || f < 0 {
		return 0, fmt.Errorf("searchParams[%q] must be a finite number >= 0, got %v", key, v)
	}
	return f, nil
}

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

// DropCollection removes the index and every indexed document hash. The ID
// counter is kept on purpose: FT.DROPINDEX DD leaves unindexed hashes behind,
// and a recreated collection that restarted at ID 1 would HSET onto them.
func (r *RedisDB) DropCollection(ctx context.Context, name string) error {
	r.mu.Lock()
	delete(r.schemas, name)
	r.mu.Unlock()
	return r.client.FTDropIndexWithArgs(ctx, name, &redis.FTDropIndexOptions{DeleteDocs: true}).Err()
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

// vectorDims returns each vector field's DIM: from the schema this process
// recorded, or (after a restart) from FT.INFO, cached for later inserts.
// No index yet means no dims to check.
func (r *RedisDB) vectorDims(ctx context.Context, collection string) (map[string]int, error) {
	schema, ok := r.schema(collection)
	if !ok {
		reply, err := r.client.Do(ctx, "FT.INFO", collection).Result()
		if err != nil {
			if msg := strings.ToLower(err.Error()); strings.Contains(msg, "unknown index") || strings.Contains(msg, "no such index") {
				return nil, nil
			}
			return nil, fmt.Errorf("redis FT.INFO %s: %w", collection, err)
		}
		attrs, _ := toMap(reply)["attributes"].([]interface{})
		for _, a := range attrs {
			m := toMap(a)
			if fmt.Sprint(m["type"]) != "VECTOR" {
				continue
			}
			dim, ok := m["dim"].(int64)
			if !ok {
				return nil, fmt.Errorf("redis FT.INFO %s: vector field %v has no integer dim", collection, m["attribute"])
			}
			schema.Fields = append(schema.Fields, Field{Name: fmt.Sprint(m["attribute"]), DataType: "float_vector", Dimension: int(dim)})
		}
		r.mu.Lock()
		if _, raced := r.schemas[collection]; !raced { // don't clobber a full schema from CreateCollection
			if r.schemas == nil {
				r.schemas = make(map[string]Schema)
			}
			r.schemas[collection] = schema
		}
		r.mu.Unlock()
	}
	dims := make(map[string]int)
	for _, f := range schema.Fields {
		if f.DataType == "float_vector" {
			dims[f.Name] = f.Dimension
		}
	}
	return dims, nil
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
	dims, err := r.vectorDims(ctx, collectionName)
	if err != nil {
		return err
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
	ef, err := efRuntime(searchParams)
	if err != nil {
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
			Params:         map[string]interface{}{"K": topK, "EF": ef, "vec": float32Bytes(vec)},
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
		id, err := keyID(doc.ID)
		if err != nil {
			return nil, err
		}
		results = append(results, SearchResult{ID: id, Score: distToScore(dist, metricType), Fields: fields})
	}
	return results, nil
}

// HybridSearch fuses BM25 text search on Text with vector KNN using FT.HYBRID.
// The query text comes from searchParams["query_text"]; when it is missing or
// matches no document, this falls back to plain KNN (see knnOnly).
// Fusion: searchParams["combine"] = "RRF" (default) or "LINEAR", with optional
// numeric "alpha"/"beta" weights (default 0.5 each). alpha, beta and "ef" are
// validated before any Redis call even when unused, so a bad config fails fast.
// FT.HYBRID takes one vector field, so several fields run one FT.HYBRID each
// (pipelined) and are merged with RRF.
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
	for field := range vectors {
		if err := validName("field", field); err != nil {
			return nil, err
		}
	}
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
	ef, err := efRuntime(searchParams)
	if err != nil {
		return nil, err
	}
	text, _ := searchParams["query_text"].(string)
	query := textQuery(text)
	// With no text, or text no document contains, FT.HYBRID would fuse KNN with an
	// arbitrary text ranking and cap every score at 0.5. Rank by vectors alone instead.
	matches, err := r.textMatches(ctx, collectionName, query)
	if err != nil {
		return nil, err
	}
	if matches == 0 {
		return r.knnOnly(ctx, collectionName, vectors, topK, metricType, searchParams)
	}
	cols := r.columns()

	pipe := r.client.Pipeline()
	cmds := make([]*redis.Cmd, 0, len(vectors))
	for field, vec := range vectors {
		cmds = append(cmds, pipe.Do(ctx, hybridArgs(collectionName, query, field, vec, topK, linear, alpha, beta, ef, cols)...))
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
		if linear {
			// LINEAR is alpha*BM25 + beta*similarity and BM25 is unbounded, so scale
			// relative to the best hit: scores land in (0,1], the top one at 1.
			top := 0.0
			for _, res := range list {
				top = math.Max(top, res.Score)
			}
			for j := range list {
				if top > 0 {
					list[j].Score /= top
				}
			}
		} else { // FT.HYBRID RRF fuses 2 lists (text, vector): max 2/(k+1). Scale into [0,1].
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

// textMatches counts documents matching the text half of a hybrid query ("*" counts as none).
func (r *RedisDB) textMatches(ctx context.Context, index, query string) (int64, error) {
	if query == "*" {
		return 0, nil
	}
	reply, err := r.client.Do(ctx, "FT.SEARCH", index, query, "LIMIT", 0, 0, "DIALECT", 2).Slice()
	if err != nil {
		return 0, fmt.Errorf("redis FT.SEARCH %s (count): %w", index, err)
	}
	if len(reply) == 0 {
		return 0, fmt.Errorf("redis FT.SEARCH %s (count): empty reply", index)
	}
	n, ok := reply[0].(int64)
	if !ok {
		return 0, fmt.Errorf("redis FT.SEARCH %s (count): unexpected total %v", index, reply[0])
	}
	return n, nil
}

// knnOnly is hybrid search without a usable text half: plain KNN per vector
// field, merged with RRF when there are several.
func (r *RedisDB) knnOnly(ctx context.Context, index string, vectors map[string]Vector, topK int, metricType string, params map[string]interface{}) ([]SearchResult, error) {
	lists := make([][]SearchResult, 0, len(vectors))
	for field, vec := range vectors {
		list, err := r.Search(ctx, index, map[string]Vector{field: vec}, topK, metricType, params)
		if err != nil {
			return nil, err
		}
		lists = append(lists, list)
	}
	if len(lists) == 1 {
		return lists[0], nil
	}
	return fuseRRF(lists, topK), nil
}

// hybridArgs builds one FT.HYBRID command (syntax: redis.io/docs/latest/commands/ft.hybrid).
func hybridArgs(index, query, field string, vec Vector, topK int, linear bool, alpha, beta float64, ef int, cols []string) []interface{} {
	window := max(topK, 20) // 20 is Redis's default fusion window
	args := []interface{}{"FT.HYBRID", index,
		"SEARCH", query,
		"VSIM", "@" + field, "$vec", "KNN", 4, "K", topK, "EF_RUNTIME", ef}
	if linear {
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
		id, err := keyID(fmt.Sprint(key))
		if err != nil {
			return nil, err
		}
		results = append(results, SearchResult{ID: id, Score: score, Fields: fields})
	}
	return results, nil
}
