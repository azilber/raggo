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
