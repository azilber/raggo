# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
go build . ./rag/... ./config/... ./examples/local_llm   # library + the local-LLM example
go vet . ./rag/... ./config/...                          # add -tags=integration to vet the integration tests too
go run ./examples/simple                                 # examples in their own dirs: simple, contextual, chat, chromem, local_llm
go run examples/full_process.go                          # loose files in examples/ are each a separate `package main`
```

- Don't run `go build ./...` / `go vet ./...`. Every loose file in `examples/` declares `package main` in one directory, so they clash. Build or run them one file at a time.
- `go.mod` requires Go 1.27.1, and anything depending on raggo inherits that floor.
- Unit tests: `go test -race . ./rag/...`. Single test: `go test ./rag -run TestName -v`.
- Redis integration tests (`rag/redis_integration_test.go`) are behind the `integration` build tag and need `REDIS_ADDR`: `docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4`, then `REDIS_ADDR=localhost:6379 go test -tags=integration -race ./rag -run TestRedis -v`.
- Most examples, and the default embedding and LLM paths, need `OPENAI_API_KEY`; other embedding servers work through `api_url` (see Embedding providers). The Milvus-backed paths need a Milvus server at `localhost:19530`.
- Never write API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) into files. Pass them through the environment only.

## End-to-end tests

These three are the required final check for work in this repo. They share one Redis: `docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4`.

1. **Gemini (automated, `rag_integration_test.go`).** It runs `RAG.LoadDocuments`, then hybrid, dense and `Retriever` queries under the default `MinScore`, then a Gemini answer about a fictional fact. It must show PASS for all 13 subtests; SKIP means the key wasn't passed.
   ```bash
   REDIS_ADDR=localhost:6379 GEMINI_API_KEY=... go test -tags=integration -race -count=1 -run TestGeminiRAG -v .
   ```
2. **llama.cpp (manual, `examples/local_llm`).** `--embeddings` restricts a server to embeddings, so run two. `/health` returns 503 while a model loads and 200 when it's ready. The chat server uses 8082 because 8080 was held by a Windows process on the WSL machine these runs were made on:
   ```bash
   llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --port 8081 &
   llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --port 8082 &
   until curl -sf localhost:8081/health && curl -sf localhost:8082/health; do sleep 2; done
   REDIS_ADDR=localhost:6379 EMBED_URL=http://localhost:8081/v1/embeddings CHAT_URL=http://localhost:8082/v1/chat/completions go run ./examples/local_llm
   pkill -x llama-server
   ```
3. **KoboldCpp (manual, `examples/local_llm`).** One process serves both endpoints on 5001. The binary (`koboldcpp-linux-x64`; use `koboldcpp-linux-x64-nocuda` without an NVIDIA GPU) and the GGUFs total about 1.7 GB, so download them into a scratch directory, never the repo:
   ```bash
   K=/tmp/koboldcpp && mkdir -p $K && cd $K
   gh release download --repo LostRuins/koboldcpp --pattern 'koboldcpp-linux-x64' --clobber && chmod +x koboldcpp-linux-x64
   curl -LO https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
   curl -LO https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf
   ./koboldcpp-linux-x64 --model gemma-3-1b-it-Q4_K_M.gguf --embeddingsmodel embeddinggemma-300M-Q8_0.gguf --port 5001 --gpulayers 99 --quiet &
   cd - && until curl -sf localhost:5001/v1/models; do sleep 2; done
   REDIS_ADDR=localhost:6379 EMBED_URL=http://localhost:5001/v1/embeddings CHAT_URL=http://localhost:5001/v1/chat/completions go run ./examples/local_llm
   pkill -f '^\./koboldcpp-linux-x64'
   ```

A llama.cpp or KoboldCpp run passes when the output shows all three of these:
- `Indexed 7 chunks (768-dim embeddings)`, which proves the index was sized from the model and not assumed to be 1536.
- `sample.txt` as the top source.
- An answer describing PressureValve scaling or load balancing. PressureValve exists only in `examples/chat/docs/sample.txt`, so a correct answer proves retrieval worked.

KoboldCpp's log shares the terminal, so `Answer:` can start mid-line. Grep for `Answer:`, not `^Answer`. KoboldCpp also takes a few seconds to exit after `pkill`.

When stopping servers, don't use `pkill -f` with a pattern that also appears in your own command line, because it kills the shell running it. Use `pkill -x llama-server`, or anchor the pattern as shown.

## Architecture

The code has two layers, with the public API on top:

- **`rag/` (low-level primitives):** the `rag.VectorDB` interface (`rag/vector_interface.go`) and its implementations (`milvus.go`, `memory.go`, `chromem.go`, `redis.go`), plus loading, parsing (PDF/txt), chunking, embedding, the sparse index, the reranker and logging.
- **Root package `raggo` (public API):** thin wrappers that use functional options (`SetX`/`WithX` funcs returning `Option`), built over `rag/`:
  - Building blocks: `loader.go`, `parser.go`, `chunker.go`, `embedder.go`, `vectordb.go`, `retriever.go`, `register.go`.
  - RAG flavors, which sit on the building blocks:
    - `SimpleRAG` (`simple_rag.go`)
    - `ContextualRAG` (`contextual_rag.go`): uses an LLM to generate per-chunk context before embedding
    - `RAG` (`rag.go`): general, option-configured
    - `MemoryContext` (`memory_context.go`): stores chat history as vectors and enhances `gollm` prompts
  - The flavors use `github.com/teilomillet/gollm` for LLM calls. `SimpleRAG`, `RAG` and `ContextualRAG` create their LLM with gollm's `openai` provider, whose endpoint is fixed at `api.openai.com` in v0.1.1. The exception is `ContextualRAG` when `ContextualRAGConfig.LLM` is set, in which case it uses that LLM.
- **`config/`:** the `config.Config` loader. Precedence is `RAGGO_*` env vars, then the JSON file (`$RAGGO_CONFIG`, `~/.raggo/config.json`, `~/.config/raggo/config.json`, `./raggo.json`), then defaults.

### Adding a vector store backend

The factory that actually runs is the `switch` in `rag.NewVectorDB` (`rag/vector_interface.go`). `raggo.NewVectorDB` (`vectordb.go`) calls it directly. The `RegisterVectorDB`/`GetVectorDB` registry in `register.go` is **not** consulted by `NewVectorDB`. A new backend needs:

1. An implementation of `rag.VectorDB` in `rag/<name>.go`.
2. A `case` in that switch.
3. Its type string documented in `WithType`.

Backend-specific settings arrive through `rag.Config.Parameters` (for example `"dimension"`).

`ContextualRAG`, `SimpleRAG` and `RAG` all take the backend from their config's `DBType`/`DBAddress` (defaults are Milvus at `localhost:19530`). `RAG`'s collection schema hardcodes `Dimension: 1536` (`rag.go`).

### Redis backend (`rag/redis.go`, Redis 8.4+)

- **Index creation:** `CreateCollection` only records the schema, and `FT.CREATE` runs in `CreateIndex`, because Redis needs the vector metric at index-creation time.
- **`searchParams`:** `query_text` (the user's text, added by `withQueryText` in `retriever.go`), `combine` (`RRF` or `LINEAR`), `alpha`/`beta` and `ef`. All of them are validated before any Redis call, even when unused, so a bad config fails fast.
- **Scores:** higher is better, and they're normalized so callers' `MinScore` filters keep working. When the text matches no document, hybrid search falls back to plain KNN; otherwise FT.HYBRID would fuse in an arbitrary text ranking.
- **Things you must preserve when changing it:**
  - Collection and field names go through `validName`, because they're pasted into query strings unparameterized.
  - `Insert` rejects vectors whose size doesn't match the index, reading the dimensions from `FT.INFO` after a restart. Redis would otherwise skip indexing them silently.
  - The ID counter `raggo:id:<col>` survives `DropCollection`, because `FT.DROPINDEX DD` leaves unindexed hashes behind that restarted IDs would overwrite.
- **Gotcha:** package `rag` defines its own `func max(a, b int) int` (`rag/chunk.go`), which shadows the builtin. Don't use `max`/`min` on non-int values there.

### Embedding providers

Providers register themselves in `init()` functions in `rag/providers/` (`openai.go`, `example_provider.go`) through `providers.Register(name, factory)`. The package keeps a separate `RegisterEmbedder` registry as well. Use `example_provider.go` as the template for a new provider.

- **Any OpenAI-compatible embeddings server** (llama.cpp, KoboldCpp, Gemini) works through the `openai` provider with `raggo.SetOption("api_url", url)` on `raggo.NewEmbedder`. The key must be non-empty even when the server ignores it.
- **A provider that needs extra request fields** (for example Gemini's `dimensions: 1536`, which makes it fit `RAG`'s fixed schema) is registered at runtime with `providers.RegisterEmbedder`, as `rag_integration_test.go` does. `RAG`, `Retriever` and `NewEmbedder` all look providers up by name.

### Local LLMs

[USAGE.md](USAGE.md) and `examples/local_llm` cover RAG on llama.cpp, KoboldCpp or Gemini. The example uses the building blocks rather than `RAG` for two reasons: `RAG`'s schema is fixed at 1536 dimensions while local embedding models are usually smaller, and gollm's `openai` endpoint can't be pointed at a local server. Keep USAGE.md's commands and sample output in sync with the example, because both were taken from real runs.
