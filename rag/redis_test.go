package rag

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
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
		// Distinct vectors: hundreds of identical points make a degenerate HNSW graph with poor recall.
		big[i] = Record{Fields: map[string]interface{}{"Embedding": []float64{0, 1, 1 + float64(i)/1000, 0}, "Text": "bulk"}}
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
