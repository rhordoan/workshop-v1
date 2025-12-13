## NIM deployment (for `module_c_rag_arch_latency_profiler.ipynb`)

The notebook's `NIMClient` expects these endpoints on a single base URL:

- `POST /v1/embeddings`
- `POST /v1/rerank`
- `POST /v1/chat/completions`

By default it uses `NIM_BASE_URL=http://localhost:8000`.

This repo includes scripts that start **3 NIM containers** (embed + rerank + chat) and an **nginx gateway** that routes the paths above to the correct backend containers.

### Prereqs

- **Docker** working (and your user in the `docker` group)
- **NVIDIA GPU** + NVIDIA container runtime working (so `docker run --gpus all ...` works)
- **NGC API key** (for pulling images from `nvcr.io`)

### 1) Export your NGC API key

Get it from NGC, then:

```bash
export NGC_API_KEY="..."
```

### 2) Start NIMs

From `fico/`:

```bash
chmod +x scripts/start_nims.sh scripts/stop_nims.sh
./scripts/start_nims.sh
```

This will expose the gateway on `http://localhost:8000`.

### 3) (Optional) Override images/models

The default models match `fico/nim_clients.py` defaults:

- `NIM_EMBED_MODEL`: `nvidia/llama-3.1-nemotron-embedding`
- `NIM_RERANK_MODEL`: `nvidia/llama-3.1-nemotron-rerank`
- `NIM_GEN_MODEL`: `meta/llama-3.1-8b-instruct`

If your NIM images/tags differ, override these before starting:

```bash
export NIM_EMBED_IMAGE="..."
export NIM_RERANK_IMAGE="..."
export NIM_GEN_IMAGE="..."
export NIM_CACHE_DIR="$HOME/.cache/nim"
```

### 4) Verify quickly

Once containers are up, the notebook should be able to hit `/v1/embeddings` without `Connection refused`.

If anything fails, check logs:

```bash
docker logs -f nim-embed
docker logs -f nim-rerank
docker logs -f nim-gen
docker logs -f nim-gateway
```

### 5) Stop

```bash
./scripts/stop_nims.sh
```




