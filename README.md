# FICO Workshop Materials (Brev-ready)

This repository contains the workshop materials under `fico/` (notebooks + helper code + a small pre-built corpus run).

## Quickstart (Brev / GPU VM)

- **Recommended**: create a **Brev Launchable** that clones this Git repo and runs the setup script.
- Brev Launchables docs: `https://docs.nvidia.com/brev/latest/launchables.html`

From a fresh machine (or in a Launchable startup script):

```bash
cd fico
./scripts/setup_workshop.sh
```

Then launch JupyterLab:

```bash
cd fico
./scripts/run_jupyter.sh
```

Open the notebooks:
- `day2_01_module_b_vector_math_rbac.ipynb`
- `day2_02_module_c_rag_arch_latency_profiler.ipynb`
- `day2_04_evals_grading_ragas.ipynb`
- `day3_02_inference_physics_nim.ipynb`
- `day3_04_observability_tracing_rag.ipynb`

## NIM mode (optional)

Module C can run in **local** mode or **NIM** mode.

To start local NIM containers + an nginx gateway on `http://localhost:8000`:

```bash
cd fico
chmod +x scripts/start_nims.sh scripts/stop_nims.sh
export NGC_API_KEY="..."   # NGC API key for nvcr.io pulls
./scripts/start_nims.sh
```

Stop them with:

```bash
cd fico
./scripts/stop_nims.sh
```

More detail: see `fico/NIM_DEPLOYMENT.md`.

## RAGAS judge (NIM) for evals

The **Evals & Grading (RAGAS)** notebook uses a **NIM chat model as the judge** via the OpenAI-compatible API:

- **`NIM_BASE_URL`**: base URL for the nginx gateway (default `http://localhost:8000`)
- **`NIM_JUDGE_MODEL`**: judge model id (defaults to `NIM_GEN_MODEL`)
- **`NIM_API_KEY`**: optional (set `NIM_API_KEY=nim` if your client library requires a value)

## Observability lab (OpenTelemetry + Jaeger)

Day 3 includes an observability notebook that uses traces to explain where time (and failures) occur in a RAG request.

Start the local tracing stack (Jaeger UI on `http://localhost:16686`):

```bash
cd fico
chmod +x scripts/start_observability.sh scripts/stop_observability.sh
./scripts/start_observability.sh
```

Stop it with:

```bash
cd fico
./scripts/stop_observability.sh
```


