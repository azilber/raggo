package raggo

import "testing"

func TestDefaultContextualConfig(t *testing.T) {
	d := DefaultContextualConfig()
	if d.DBType != "milvus" || d.DBAddress != "localhost:19530" {
		t.Errorf("contextual defaults changed: %s %s", d.DBType, d.DBAddress)
	}
}
