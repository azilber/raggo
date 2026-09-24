# Local RAG with raggo, Redis, and llama.cpp or KoboldCpp

raggo parses and chunks your documents, embeds them through a local OpenAI-compatible `/v1/embeddings` endpoint, and stores and retrieves them in Redis with hybrid (BM25 + vector) search. A local chat model then answers from the retrieved chunks. Nothing leaves your machine.

The complete program is [`examples/local_llm/main.go`](examples/local_llm/main.go).

## Requirements

- Go 1.27.1+
- Redis 8.4+ (for `FT.HYBRID`): `docker run -d --rm -p 6379:6379 --name raggo-redis redis:8.4`
- One of the servers below, plus an embeddings GGUF and a chat GGUF. The examples use [`ggml-org/embeddinggemma-300M-GGUF`](https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF) (768-dim) and [`ggml-org/gemma-3-1b-it-GGUF`](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF).

## Start the model server

### llama.cpp

A server started with `--embeddings` serves embeddings only, so run two. `-hf` downloads the model on first use:

```bash
llama-server -hf ggml-org/embeddinggemma-300M-GGUF --embeddings --pooling mean --port 8081
llama-server -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --port 8080
```

```bash
export EMBED_URL=http://localhost:8081/v1/embeddings
export CHAT_URL=http://localhost:8080/v1/chat/completions
```

Both servers are ready when `curl localhost:8081/health` and `curl localhost:8080/health` return 200.

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

It is ready when `curl localhost:5001/v1/models` returns 200. (Tested with the `koboldcpp-linux-x64` v1.121 release on an NVIDIA GPU; the same release also ships a `koboldcpp-linux-x64-nocuda` build.)

## Run

```bash
REDIS_ADDR=localhost:6379 go run ./examples/local_llm -docs examples/chat/docs -q "What did the PressureValve system do during Black Friday?"
```

Output with either server and the models above:

```text
Indexed 7 chunks (768-dim embeddings)
Inserting 7 records into collection: local_docs
Performing hybrid search in collection local_docs for top 3 results with metric type COSINE
Sources:
  1.000 {"source":"sample.txt"}
  0.984 {"source":"microservices.txt"}
  0.968 {"source":"microservices.txt"}
Answer: During Black Friday, the PressureValve system automatically scaled resources based on incoming traffic patterns and distributed load across multiple servers, successfully maintaining system stability and ensuring zero downtime by dynamically allocating resources and routing requests efficiently.
```

PressureValve exists only in `examples/chat/docs/sample.txt`, so the answer comes from retrieval, not from the model's training data.

## How it hooks up

**Embeddings from the local server.** raggo's `openai` embedder works with any OpenAI-compatible server when you set `api_url`. Local servers ignore the model and key, but raggo requires a non-empty key:

```go
embedder, err := raggo.NewEmbedder(
	raggo.SetEmbedderProvider("openai"),
	raggo.SetEmbedderModel("local"),
	raggo.SetEmbedderAPIKey("none"),
	raggo.SetOption("api_url", embedURL),
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

**Answer:** POST the retrieved chunks and the question to `CHAT_URL` (`/v1/chat/completions`) and read `choices[0].message.content`. See `chat` in the example.

## Why not `raggo.RAG` or `raggo.SimpleRAG`?

- `RAG`'s collection schema is fixed at 1536 dimensions. Most local embedding models use 384–1024, so every insert would be rejected.
- raggo's built-in LLM calls go through gollm v0.1.1, which only targets `api.openai.com`.

The building blocks above avoid both limits.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `vector has N dimensions, index expects M` | The collection was created with a different embedding model. The example recreates `local_docs` on every run; in your own code, drop or rename the collection when you change models. |
| `ERR unknown command 'FT.HYBRID'` | Redis is older than 8.4. |
| `embeddings endpoint ...: connection refused` | The server is still loading the model; wait for `/health` (llama.cpp) or `/v1/models` (KoboldCpp). |
| `couldn't bind HTTP server socket` (llama.cpp) | The port is taken, possibly by a Windows process under WSL. Pick another `--port` and update `CHAT_URL` or `EMBED_URL`. |
