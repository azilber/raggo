package raggo

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

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
}

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
