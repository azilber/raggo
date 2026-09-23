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
