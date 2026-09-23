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
