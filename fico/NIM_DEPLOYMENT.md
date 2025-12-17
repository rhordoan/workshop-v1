## Local LLM deployment (recommended): Ollama

This repo is configured to use a **local Ollama server** (OpenAI-compatible) for chat/completions.

### Endpoint

- **Base URL**: `http://localhost:11434`
- **Chat**: `POST /v1/chat/completions`
- **Model** (default): `llama3.1:8b`

### Quick verify (curl)

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "llama3.1:8b",
        "messages": [{"role":"user","content":"Hello!"}],
        "max_tokens": 64,
        "temperature": 0.0
      }'
```

### Start

From `fico/`:

```bash
chmod +x scripts/start_ollama.sh scripts/stop_ollama.sh
./scripts/start_ollama.sh
```

### Stop

```bash
./scripts/stop_ollama.sh
```

### Notes

- Some notebooks still use env var names like `NIM_BASE_URL` / `NIM_GEN_MODEL` for compatibility; point them at Ollama:
  - `NIM_BASE_URL=http://localhost:11434`
  - `NIM_GEN_MODEL=llama3.1:8b`




