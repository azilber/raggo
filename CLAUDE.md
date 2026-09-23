# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
go build . ./rag/... ./config/...   # build the library
go vet . ./rag/... ./config/...
go run ./examples/simple            # examples in their own dirs: simple, contextual, chat, chromem
go run examples/full_process.go     # loose files in examples/ are each a separate `package main`
```

- Don't run `go build ./...` / `go vet ./...`. Every loose file in `examples/` declares `package main` in one directory, so they clash. Build or run them one file at a time.
- There are no `_test.go` files yet. To run a single test once some exist: `go test ./rag -run TestName -v`.
- Most examples and all LLM or embedding paths need `OPENAI_API_KEY`. The Milvus-backed paths need a Milvus server at `localhost:19530`.

## Architecture

The code has two layers, with the public API on top:

- **`rag/` (low-level primitives):** the `rag.VectorDB` interface (`rag/vector_interface.go`) and its implementations (`milvus.go`, `memory.go`, `chromem.go`), plus loading, parsing (PDF/txt), chunking, embedding, the sparse index, the reranker and logging.
- **Root package `raggo` (public API):** thin wrappers that use functional options (`SetX`/`WithX` funcs returning `Option`), built over `rag/`:
  - Building blocks: `loader.go`, `parser.go`, `chunker.go`, `embedder.go`, `vectordb.go`, `retriever.go`, `register.go`.
  - RAG flavors, which sit on the building blocks:
    - `SimpleRAG` (`simple_rag.go`)
    - `ContextualRAG` (`contextual_rag.go`): uses an LLM to generate per-chunk context before embedding
    - `RAG` (`rag.go`): general, option-configured
    - `MemoryContext` (`memory_context.go`): stores chat history as vectors and enhances `gollm` prompts
  - The flavors use `github.com/teilomillet/gollm` for LLM calls.
- **`config/`:** the `config.Config` loader. Precedence is `RAGGO_*` env vars, then the JSON file (`$RAGGO_CONFIG`, `~/.raggo/config.json`, `~/.config/raggo/config.json`, `./raggo.json`), then defaults.

### Adding a vector store backend

The factory that actually runs is the `switch` in `rag.NewVectorDB` (`rag/vector_interface.go`). `raggo.NewVectorDB` (`vectordb.go`) calls it directly. The `RegisterVectorDB`/`GetVectorDB` registry in `register.go` is **not** consulted by `NewVectorDB`. A new backend needs:

1. An implementation of `rag.VectorDB` in `rag/<name>.go`.
2. A `case` in that switch.
3. Its type string documented in `WithType`.

Backend-specific settings arrive through `rag.Config.Parameters` (for example `"dimension"`).

`ContextualRAG`, `SimpleRAG` and `RAG` all take the backend from their config's `DBType`/`DBAddress` (defaults are Milvus at `localhost:19530`).

Redis (`rag/redis.go`) needs Redis 8.4+ (`FT.HYBRID`). Its hybrid search reads the user's text from `searchParams["query_text"]`, which `withQueryText` (`retriever.go`) adds. Integration tests run with `REDIS_ADDR=localhost:6379 go test ./rag -run TestRedis -v` against `docker run -p 6379:6379 redis:8.4`.

### Embedding providers

Providers register themselves in `init()` functions in `rag/providers/` (`openai.go`, `example_provider.go`) through `providers.Register(name, factory)`. The package keeps a separate `RegisterEmbedder` registry as well. Use `example_provider.go` as the template for a new provider.
