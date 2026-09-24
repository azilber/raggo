# RAG with raggo and Redis: llama.cpp, KoboldCpp, or Gemini

raggo parses and chunks your documents, embeds them through an OpenAI-compatible `/v1/embeddings` endpoint, and stores and retrieves them in Redis with hybrid (BM25 + vector) search. A chat model then answers from the retrieved chunks. With llama.cpp or KoboldCpp everything runs on your machine; with Gemini the embedding and chat calls go to Google's API.

The complete program is [`examples/local_llm/main.go`](examples/local_llm/main.go). The same binary works with all three backends; only environment variables change.

## Requirements

- Go 1.27.1+
- Redis 8.4+ (for `FT.HYBRID`): `docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4`
- One backend:
  - **llama.cpp or KoboldCpp**, plus an embeddings GGUF and a chat GGUF. The examples use [`ggml-org/embeddinggemma-300M-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) (768-dim) and [`ggml-org/gemma-3-1b-it-GGUF`](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF).
  - **Gemini**: a Gemini API key.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_ADDR` | `localhost:6379` | Redis address, or a `redis://user:pass@host:port/db` URL |
| `EMBED_URL` | `http://localhost:8081/v1/embeddings` | OpenAI-compatible embeddings endpoint |
| `CHAT_URL` | `http://localhost:8080/v1/chat/completions` | OpenAI-compatible chat endpoint |
| `API_KEY` | `none` | Sent as `Authorization: Bearer …`; local servers ignore it |
| `EMBED_MODEL` | `local` | Embeddings model name; local servers ignore it |
| `CHAT_MODEL` | *(empty, not sent)* | Chat model name; required by hosted APIs such as Gemini |

## Start a backend

### llama.cpp

A server started with `--embeddings` serves embeddings only, so run two. `-hf` downloads the model on first use. `-ngl 99` offloads layers to the GPU only if your build has a GPU backend; check with `llama-server --list-devices` (a CPU-only build, such as the Homebrew one, lists only `BLAS` and ignores `-ngl`):

```bash
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --port 8081
llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --port 8080
```

```bash
export EMBED_URL=http://localhost:8081/v1/embeddings
export CHAT_URL=http://localhost:8080/v1/chat/completions
```

Both are ready when `curl localhost:8081/health` and `curl localhost:8080/health` return 200.

**CPU only.** Drop `-ngl` and add `--device none`, which tells a GPU build not to offload (a CPU-only build runs on CPU either way):

```bash
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --device none --port 8081
llama-server -hf ggml-org/gemma-3-1b-it-GGUF --device none --port 8080
```

### KoboldCpp

One process serves both, on port 5001. KoboldCpp loads local GGUF files, so download them first:

```bash
curl -LO https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf
curl -LO https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300M-Q8_0.gguf
./koboldcpp-linux-x64 --model gemma-3-1b-it-Q4_K_M.gguf --embeddingsmodel embeddinggemma-300M-Q8_0.gguf --port 5001 --gpulayers 99
```

```bash
export EMBED_URL=http://localhost:5001/v1/embeddings
export CHAT_URL=http://localhost:5001/v1/chat/completions
```

It is ready when `curl localhost:5001/v1/models` returns 200. Tested with the `koboldcpp-linux-x64` v1.121 release on an NVIDIA GPU.

**CPU only.** Either add `--usecpu` to the CUDA build and drop `--gpulayers`, or use the smaller `koboldcpp-linux-x64-nocuda` build (from the same release) with no GPU flags:

```bash
./koboldcpp-linux-x64 --usecpu --model gemma-3-1b-it-Q4_K_M.gguf --embeddingsmodel embeddinggemma-300M-Q8_0.gguf --port 5001
./koboldcpp-linux-x64-nocuda --model gemma-3-1b-it-Q4_K_M.gguf --embeddingsmodel embeddinggemma-300M-Q8_0.gguf --port 5001
```

To confirm where the model runs, read the startup log's buffer lines: CPU runs show `CPU model buffer size`, GPU runs show `CUDA0 model buffer size`. With `--usecpu` the log still prints `offloaded 27/27 layers to GPU`, but the buffers are all `CPU`; the `-nocuda` build prints `offloaded 0/27`.

### Gemini

Gemini serves OpenAI-compatible endpoints, so no server is needed. Unlike the local servers, it requires the key and both model names:

```bash
export API_KEY=$GEMINI_API_KEY
export EMBED_URL=https://generativelanguage.googleapis.com/v1beta/openai/embeddings
export EMBED_MODEL=gemini-embedding-001
export CHAT_URL=https://generativelanguage.googleapis.com/v1beta/openai/chat/completions
export CHAT_MODEL=gemini-3.6-flash
```

`gemini-embedding-001` returns 3072-dimension vectors; the program sizes the Redis index from the model, so nothing else changes.

## Run

```bash
REDIS_ADDR=localhost:6379 go run ./examples/local_llm -docs examples/chat/docs -q "What did the PressureValve system do during Black Friday?"
```

Each run replaces the `local_docs` index in Redis. It only drops the old index after every document has been parsed and embedded, so a wrong `-docs` path or a failing endpoint leaves it intact.

Output with llama.cpp or KoboldCpp and the models above:

```text
Inserting 7 records into collection: local_docs
Indexed 7 chunks (768-dim embeddings)
Performing hybrid search in collection local_docs for top 3 results with metric type COSINE
Sources:
  1.000 {"source":"sample.txt"}
  0.984 {"source":"microservices.txt"}
  0.968 {"source":"microservices.txt"}
Answer: During Black Friday, the PressureValve system automatically scaled resources based on incoming traffic patterns and distributed load across multiple servers, successfully maintaining system stability and ensuring zero downtime by dynamically allocating resources and routing requests efficiently.
```

With Gemini the index line reads `Indexed 7 chunks (3072-dim embeddings)`, the same three sources come back, and the answer is a longer bulleted summary of the same document.

PressureValve exists only in `examples/chat/docs/sample.txt`, so the answer comes from retrieval, not from the model's training data.

### CPU vs GPU

Every llama.cpp and KoboldCpp setup above, CPU or GPU, produced the output shown. Wall time for `go run ./examples/local_llm` (7 chunks, one answer, compile included), one run each on an AMD Ryzen 9 5950X (16 cores) and an RTX 3080:

| Setup | Wall time |
|---|---|
| KoboldCpp, CUDA build, `--gpulayers 99` | 2.4 s |
| KoboldCpp, CUDA build, `--usecpu` | 5.0 s |
| KoboldCpp, `-nocuda` build | 4.9 s |
| llama.cpp, CPU (`--device none`, Homebrew build) | 11.3 s |

## How it hooks up

**Embeddings from any OpenAI-compatible endpoint.** raggo's `openai` embedder works with any such server when you set `api_url`. raggo requires a non-empty key even when the server ignores it:

```go
embedder, err := raggo.NewEmbedder(
	raggo.SetEmbedderProvider("openai"),
	raggo.SetEmbedderModel(cfg.embedModel),
	raggo.SetEmbedderAPIKey(cfg.apiKey),
	raggo.SetOption("api_url", cfg.embedURL),
)
```

**Redis index sized to the model.** Embed a probe string and use its length as the vector dimension. Redis silently skips indexing any vector whose size differs from the index, so raggo's `Insert` rejects the mismatch instead:

```go
probe, err := embedder.Embed(ctx, "dimension probe")
// ...
{Name: "Embedding", DataType: "float_vector", Dimension: len(probe)},
```

**Chunk, embed, store:** `raggo.NewParser().Parse(file)`, then `raggo.NewChunker(raggo.ChunkSize(200), raggo.ChunkOverlap(50))`, `embedder.Embed` for each chunk, and `db.Insert` with `Embedding`, `Text` and `Metadata` fields.

**Retrieve:** hybrid search fuses BM25 on `Text` with vector similarity. Pass the raw question as `query_text`:

```go
results, err := db.HybridSearch(ctx, collection, map[string]raggo.Vector{"Embedding": qvec}, 3, "COSINE",
	map[string]interface{}{"query_text": question, "ef": 64}, nil)
```

If no document contains any word of the question, this falls back to vector-only search.

**Answer:** POST the retrieved chunks and the question to `CHAT_URL` (`/v1/chat/completions`) with `Authorization: Bearer $API_KEY` (and `model` when `CHAT_MODEL` is set), then read `choices[0].message.content`. See `chat` in the example.

## Using `raggo.RAG` with Gemini

`raggo.RAG` can also run on Gemini and Redis, because Gemini's embeddings API accepts a `dimensions` parameter that matches `RAG`'s fixed 1536-dimension schema. raggo's built-in `openai` embedder can't send that parameter, so register your own provider and point `RAG` at it:

```go
providers.RegisterEmbedder("gemini", func(cfg map[string]interface{}) (providers.Embedder, error) {
	apiKey, _ := cfg["api_key"].(string)
	model, _ := cfg["model"].(string)
	return geminiEmbedder{apiKey: apiKey, model: model}, nil // POSTs {"model", "input", "dimensions": 1536}
})

r, err := raggo.NewRAG(
	raggo.SetProvider("gemini"),
	raggo.SetModel("gemini-embedding-001"),
	raggo.SetAPIKey(os.Getenv("GEMINI_API_KEY")),
	raggo.SetDBType("redis"),
	raggo.SetDBAddress("localhost:6379"),
	raggo.SetCollection("docs"),
)
err = r.LoadDocuments(ctx, "examples/chat/docs")
results, err := r.Query(ctx, "What did the PressureValve system do during Black Friday?")
```

`providers` is `github.com/teilomillet/raggo/rag/providers`. The full `geminiEmbedder`, a retry on HTTP 429/5xx, the chat call, and assertions over hybrid, dense and `Retriever` queries are in [`rag_integration_test.go`](rag_integration_test.go):

```bash
REDIS_ADDR=localhost:6379 GEMINI_API_KEY=... go test -tags=integration -run TestGeminiRAG -v .
```

## Why not `raggo.RAG` with llama.cpp or KoboldCpp?

- `RAG`'s collection schema is fixed at 1536 dimensions (`rag.go`). Most local embedding models output 384–1024 (embeddinggemma-300M: 768), so every insert would be rejected.
- raggo's built-in LLM calls use gollm's `openai` provider, whose endpoint is fixed at `api.openai.com` in gollm v0.1.1, so they can't reach a local chat server.

The building blocks in `examples/local_llm` avoid both limits.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `vector has N dimensions, index expects M` | The collection was created with a different embedding model. The example recreates `local_docs` on every run; in your own code, drop or rename the collection when you change models. |
| `unknown command 'FT.HYBRID'` or `unknown command 'FT.INFO'` | Redis is older than 8.4 (`FT.INFO` missing means no query engine at all). |
| `API request failed with status code 503: 503 Service Unavailable` (llama.cpp) | The server is up but still loading the model; wait for `/health` to return 200. |
| `embeddings endpoint ...: connection refused` | The server isn't running yet, or `EMBED_URL` has the wrong port. |
| `-ngl 99` makes no difference (llama.cpp) | Your build has no GPU backend: `llama-server --list-devices` lists only `BLAS`/CPU. It runs on CPU; install a CUDA/Vulkan/Metal build for GPU offload. |
| `couldn't bind HTTP server socket` (llama.cpp) | The port is taken, possibly by a Windows process under WSL. Pick another `--port` and update `CHAT_URL` or `EMBED_URL`. |
| `HTTP 404` from the chat endpoint (Gemini) | `CHAT_MODEL` names a model your key can't use; check the error message for the suggested model. |
