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
}
