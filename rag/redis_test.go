package rag

import (
	"context"
	"encoding/json"
	"math"
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
		{name: "zero is allowed", params: map[string]interface{}{"alpha": 0}, want: 0},
		{name: "uint", params: map[string]interface{}{"alpha": uint(1)}, want: 1},
		{name: "int8", params: map[string]interface{}{"alpha": int8(2)}, want: 2},
		{name: "uint64", params: map[string]interface{}{"alpha": uint64(3)}, want: 3},
		{name: "json.Number from UseNumber", params: map[string]interface{}{"alpha": json.Number("0.25")}, want: 0.25},
		{name: "malformed json.Number", params: map[string]interface{}{"alpha": json.Number("x")}, wantErr: true},
		{name: "NaN is an error", params: map[string]interface{}{"alpha": math.NaN()}, wantErr: true},
		{name: "+Inf is an error", params: map[string]interface{}{"alpha": math.Inf(1)}, wantErr: true},
		{name: "negative is an error", params: map[string]interface{}{"alpha": -1}, wantErr: true},
		{name: "nil value is an error", params: map[string]interface{}{"alpha": nil}, wantErr: true},
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
	tests := []struct {
		name string
		addr string
	}{
		// url.Parse's own error quotes the whole URL, password included.
		{name: "unparseable URL", addr: "redis://user:s3cret@host:notaport/0"},
		{name: "unreachable server", addr: "redis://user:s3cret@127.0.0.1:1/0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
			defer cancel()
			db, _ := newRedisDB(&Config{Address: tt.addr})
			err := db.Connect(ctx)
			if err == nil {
				t.Fatal("expected a connection error")
			}
			if strings.Contains(err.Error(), "s3cret") {
				t.Errorf("error leaks password: %v", err)
			}
		})
	}
}

// A bad LINEAR weight must fail before any Redis call: this RedisDB was never
// connected, so reaching Redis would nil-deref instead of returning an error.
func TestRedisHybridSearchRejectsBadWeightBeforeRedis(t *testing.T) {
	db, _ := newRedisDB(&Config{})
	_, err := db.HybridSearch(context.Background(), "docs", map[string]Vector{"Embedding": {1, 0}}, 3, "COSINE",
		map[string]interface{}{"combine": "LINEAR", "alpha": "0.3"}, nil)
	if err == nil || !strings.Contains(err.Error(), "alpha") {
		t.Errorf("err = %v, want error naming alpha", err)
	}
}
