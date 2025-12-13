from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterSpec:
    name: str
    owner_teams: list[str]
    doc_type_weights: dict[str, float]
    base_tags: list[str]
    # LLM prompt "topic anchors" for this cluster.
    topic_hints: list[str]


CLUSTERS: list[ClusterSpec] = [
    ClusterSpec(
        name="Finance",
        owner_teams=["credit-risk", "underwriting", "fraud"],
        doc_type_weights={"policy": 0.35, "kb": 0.45, "incident": 0.05, "runbook": 0.15},
        base_tags=["fico", "credit", "apr", "loan", "dti", "interest-rate", "underwriting"],
        topic_hints=[
            "credit scoring factors and model governance",
            "APR calculation and interest rate adjustments",
            "loan default risk and debt-to-income policy thresholds",
            "FICO score versions and underwriting guidance",
        ],
    ),
    ClusterSpec(
        name="Infrastructure",
        owner_teams=["sre", "platform", "security"],
        doc_type_weights={"runbook": 0.40, "incident": 0.40, "kb": 0.15, "policy": 0.05},
        base_tags=["kubernetes", "gpu", "latency", "firewall", "docker", "driver", "observability"],
        topic_hints=[
            "Kubernetes incident response and crash loops",
            "GPU memory failures and inference performance regressions",
            "latency spikes, regional outages, and capacity planning",
            "driver/CUDA compatibility and container deployment issues",
        ],
    ),
    ClusterSpec(
        name="Noise",
        owner_teams=["public-web"],
        doc_type_weights={"public_summary": 0.70, "kb": 0.20, "policy": 0.05, "incident": 0.05},
        base_tags=["food", "weather", "sports", "travel"],
        topic_hints=[
            "recipes and cooking tips",
            "weather summaries and travel notes",
            "sports commentary and match results",
        ],
    ),
]


_CLEAN_RE = re.compile(r"[\r\n\t]+")


def _clean_line(s: str) -> str:
    s = _CLEAN_RE.sub(" ", s).strip()
    s = re.sub(r"\s{2,}", " ", s)
    # Strip common "assistant:" style prefixes if they appear.
    s = re.sub(r"^(assistant|response)\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    # Remove leading list markers if model ignores instructions.
    s = re.sub(r"^\s*[-*]\s*", "", s).strip()
    s = re.sub(r"^\s*\d+\.\s*", "", s).strip()
    return s


def _try_cuda_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _extract_json_object(s: str) -> dict | None:
    """
    Best-effort extraction of the first JSON object in a string.
    """
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    snippet = s[start : end + 1]
    try:
        return json.loads(snippet)
    except Exception:
        return None


def generate_texts_with_llm(
    model_id: str,
    prompts: list[str],
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
) -> list[str]:
    """
    Returns: one generated string per prompt.
    Uses GPU (A100) automatically when CUDA is available.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    # Decoder-only models should be left-padded for batched generation.
    # This avoids warnings and prevents degraded generations due to right-padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.backends.cuda.matmul.allow_tf32 = True

    # transformers deprecated torch_dtype= in favor of dtype=
    dtype = torch.bfloat16 if torch.cuda.is_available() else None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    except TypeError:
        # Back-compat with older transformers versions
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
    model.eval()

    def _format_prompt(p: str) -> str:
        # If tokenizer supports chat templates, use them; otherwise plain prompt.
        if hasattr(tokenizer, "apply_chat_template"):
            msgs = [{"role": "user", "content": p}]
            return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return p + "\n"

    formatted = [_format_prompt(p) for p in prompts]

    results: list[str] = []
    for i in range(0, len(formatted), batch_size):
        batch_prompts = formatted[i : i + batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated portion (avoid including prompt).
        attn = inputs.get("attention_mask")
        if attn is None:
            prompt_lens = [inputs["input_ids"].shape[1]] * out.shape[0]
        else:
            prompt_lens = attn.sum(dim=1).tolist()

        for seq, plen in zip(out, prompt_lens):
            gen_ids = seq[int(plen) :]
            results.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

    return results


def _sentences(s: str) -> list[str]:
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    # A very simple sentence splitter; good enough for synthetic corpora.
    parts = re.split(r"(?<=[.!?])\s+", s)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def _ensure_sentence_count(body: str, min_s: int = 3, max_s: int = 10) -> str:
    sents = _sentences(body)
    if not sents:
        return body
    if len(sents) < min_s:
        # Repeat/extend minimally by duplicating the last sentence with small variation.
        while len(sents) < min_s:
            sents.append(sents[-1])
    if len(sents) > max_s:
        sents = sents[:max_s]
    return " ".join(sents)


def _redact_body(body: str) -> tuple[str, int]:
    """
    Redact lines that look sensitive. Returns (redacted_body, redaction_count).
    """
    redactions = 0
    lines = []
    for line in body.splitlines():
        if any(line.strip().startswith(prefix) for prefix in ("SECRET:", "PII:", "CREDENTIAL:", "TOKEN:")):
            lines.append("[REDACTED]")
            redactions += 1
        else:
            lines.append(line)
    return "\n".join(lines), redactions


def _doc_text(title: str, body: str, tags: list[str], doc_type: str, tenant_id: str) -> str:
    # One "text" field for embedding + naive keyword search.
    tag_str = ", ".join(tags)
    return f"TITLE: {title}\nTYPE: {doc_type}\nTENANT: {tenant_id}\nTAGS: {tag_str}\n\n{body}".strip()


def _embed_text(
    title: str,
    body: str,
    tags: list[str],
    doc_type: str,
    tenant_id: str,
    max_chars: int = 2000,
) -> str:
    """
    Build a bounded text used for embeddings to avoid exceeding model max sequence length.
    Keeps the most salient metadata + a truncated body.
    """
    tag_str = ", ".join(tags[:12])
    head = f"TITLE: {title}\nTYPE: {doc_type}\nTENANT: {tenant_id}\nTAGS: {tag_str}\n\n"
    remaining = max(0, max_chars - len(head))
    body_trim = (body or "").strip()
    if remaining and len(body_trim) > remaining:
        body_trim = body_trim[:remaining].rsplit(" ", 1)[0].strip() + "…"
    return (head + body_trim).strip()


def _rbac_policy_for_doc(
    doc_type: str,
    tenant_id: str,
    tags: list[str],
    access_level: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Returns (allowed_roles, allowed_tenants, restricted_tags).
    restricted_tags are tags that require explicit clearance.
    """
    # Tenant scope
    allowed_tenants = ["*"] if tenant_id == "global" else [tenant_id]

    # Roles
    if doc_type == "public_summary" or access_level == 1:
        allowed_roles = ["public"]
    elif doc_type in {"incident", "runbook"}:
        allowed_roles = ["sre", "admin", "security"]
    elif doc_type in {"policy", "kb"}:
        allowed_roles = ["analyst", "risk_analyst", "admin", "security"]
    else:
        allowed_roles = ["admin", "security"]

    # Tag-based restrictions (must have clearance)
    restricted_vocab = {"pii", "secrets", "customer-data", "prod", "credentials"}
    restricted_tags = sorted(list(set(tags).intersection(restricted_vocab)))
    if access_level >= 3 and not restricted_tags:
        # Admin-level docs often have something to restrict.
        restricted_tags = ["secrets"]

    return allowed_roles, allowed_tenants, restricted_tags


def _infer_access_level(doc_type: str, tags: list[str]) -> int:
    if doc_type == "public_summary":
        return 1
    if any(t in {"secrets", "pii", "credentials"} for t in tags):
        return 3
    return 2


def _build_doc_prompt(
    cluster: ClusterSpec,
    topic: str,
    doc_type: str,
    overlap_hint: str | None,
    force_public: bool,
    force_internal: bool,
) -> str:
    """
    Ask the LLM for a JSON object with multi-field doc content.
    """
    tag_guidance = ", ".join(cluster.base_tags[:8])
    overlap = ""
    if overlap_hint:
        overlap = f"\nAlso include this cross-domain detail: {overlap_hint}"

    if force_public:
        sensitivity = "This is PUBLIC, safe to share externally. Avoid secrets and PII."
    elif force_internal:
        sensitivity = (
            "This is INTERNAL and sensitive. Include 1-2 redactable lines prefixed with 'SECRET:' or 'PII:'."
        )
    else:
        sensitivity = "This is internal. Avoid PII. If mentioning sensitive details, use 'SECRET:' lines."

    return (
        f"Create ONE {doc_type} document.\n"
        f"Topic: {topic}\n"
        f"{sensitivity}\n"
        "Write a TITLE and a BODY of 3-10 sentences.\n"
        f"Include 3-7 tags, preferably from: {tag_guidance}\n"
        f"{overlap}\n\n"
        "Return ONLY valid JSON with this schema:\n"
        '{ "title": "...", "body": "...", "tags": ["...", "..."] }'
    )


def synthesize_texts_template(n: int, seed: int) -> list[list[str]]:
    """
    Fallback generator that does not require an LLM.
    Returns list per cluster prompt.
    """
    rng = np.random.default_rng(seed)

    finance = [
        "APR calculation changes for variable-rate credit cards this quarter.",
        "Credit score impact increases with high utilization above thirty percent.",
        "Loan default probability rises when DTI exceeds forty percent.",
        "Interest rate adjustment depends on prime rate movement and borrower risk.",
        "FICO Score 8 differs from 9 in medical debt treatment.",
        "Debt-to-income ratio is a key underwriting constraint for mortgages.",
    ]
    infra = [
        "Kubernetes pods crashloop after the sidecar injection policy update.",
        "GPU memory overflow occurs during large-batch embedding inference on A100.",
        "Latency spikes in us-east correlate with overloaded ingress controllers.",
        "Firewall rules block outbound traffic to the model registry endpoint.",
        "Docker containers restart repeatedly after missing environment secrets.",
        "NVIDIA driver updates can break CUDA compatibility for inference workloads.",
    ]
    noise = [
        "Pepperoni pizza tastes best with fresh basil and hot honey drizzle.",
        "Bake the cake until a toothpick comes out clean from the center.",
        "Spicy tacos pair well with pickled onions and lime crema.",
        "Weather in London is often rainy with sudden sunny intervals.",
        "Football match results vary widely depending on home field advantage.",
        "Organic apple pie is great with cinnamon and a flaky crust.",
    ]

    pools = [finance, infra, noise]
    return [[rng.choice(pool) for _ in range(n)] for pool in pools]


def embed_texts(
    texts: Iterable[str],
    embed_model_id: str,
    batch_size: int,
    seed: int,
    max_tokens: int = 480,
) -> np.ndarray:
    try:
        import torch
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "PyTorch is required for embedding. Install a CUDA-enabled build for A100, e.g.:\n"
            "  - follow https://pytorch.org/get-started/locally/\n"
            "Then re-run this script.\n"
            "If you just want runnable demo coordinates without embeddings, pass --pseudo-embed."
        ) from e

    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    model = SentenceTransformer(embed_model_id, device=device)
    # Ensure SentenceTransformer truncates during its internal tokenize() step.
    # Without this, very long inputs can trigger "Token indices sequence length..." warnings.
    try:
        model.max_seq_length = int(max_tokens)
    except Exception:
        pass

    # Hard truncate by tokens to avoid exceeding the underlying model's max sequence length.
    # (Char-based trimming is not sufficient because tokenization varies.)
    try:
        tok = model.tokenizer
        enc = tok(
            list(texts),
            truncation=True,
            max_length=int(max_tokens),
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        truncated_texts = [tok.decode(ids, skip_special_tokens=True) for ids in enc["input_ids"]]
    except Exception:
        truncated_texts = list(texts)

    emb = model.encode(
        truncated_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return emb


def _write_json(path: str, obj: object) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic FICO-ish corpus on GPU (A100) and embed it for day2_01_module_b_vector_math_rbac.ipynb."
    )
    parser.add_argument(
        "--out",
        default="fico_corpus_embedded.csv",
        help="Output CSV path (embedded x/y/z). If --out-dir is set, this is treated as a filename.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Optional output directory. If set, all artifacts are written there.",
    )
    parser.add_argument(
        "--prefix",
        default="fico_corpus",
        help="Filename prefix for extra artifacts (jsonl/txt/stats) when --out-dir is set.",
    )
    parser.add_argument("--n-per-cluster", type=int, default=250, help="Documents per cluster.")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--gen-model",
        default=os.environ.get("FICO_GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        help="HF model id for text generation (set env FICO_GEN_MODEL to override).",
    )
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("FICO_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        help="SentenceTransformer model id (set env FICO_EMBED_MODEL to override).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gen-batch-size", type=int, default=8)
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument(
        "--embed-max-tokens",
        type=int,
        default=480,
        help="Max tokens for embed_text truncation before embedding (prevents max-seq warnings).",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Write <prefix>_embeddings.npy and <prefix>_pca.json for real vector search + query projection.",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Do not use an LLM; generate from templates only.",
    )
    parser.add_argument(
        "--overlap-rate",
        type=float,
        default=0.15,
        help="Fraction of docs that intentionally mix cross-domain vocabulary (0..1).",
    )
    parser.add_argument(
        "--pair-rate",
        type=float,
        default=0.10,
        help="Fraction of docs per cluster emitted as public/internal near-duplicate pairs (0..1).",
    )
    parser.add_argument(
        "--tenants",
        default="acme,globex,initech",
        help="Comma-separated tenant ids used for tenant_id scoping; 'global' is added automatically.",
    )
    parser.add_argument(
        "--pseudo-embed",
        action="store_true",
        help="Do not compute real embeddings; generate synthetic x/y/z coordinates (demo-only).",
    )
    parser.add_argument(
        "--save-corpus",
        action="store_true",
        help="Write raw corpus artifacts: <prefix>.jsonl and <prefix>.txt (one text per line).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Write <prefix>_stats.json and <prefix>_sizes.txt with counts and file sizes.",
    )

    args = parser.parse_args()

    device = _try_cuda_device()
    if not args.template_only and device != "cuda":
        print("⚠️ CUDA not detected; falling back to template-only generation.")
        args.template_only = True

    n = int(args.n_per_cluster)
    seed = int(args.seed)

    print(f"✅ Target: {n} docs/cluster, seed={seed}")

    if args.template_only:
        print("⚙️ Generating texts from templates (no LLM).")
        per_cluster_texts = synthesize_texts_template(n=n, seed=seed)
    else:
        print(f"⚙️ Using CUDA GPU for generation with model: {args.gen_model}")

    rng = np.random.default_rng(seed)
    tenants = [t.strip() for t in str(args.tenants).split(",") if t.strip()]
    tenants = sorted(list(dict.fromkeys(tenants + ["global"])))

    # Build per-document prompts and metadata so we can create multi-field docs.
    overlap_rate = float(args.overlap_rate)
    pair_rate = float(args.pair_rate)

    overlap_hints = [
        "GPU cost impacts APR model training latency under peak load",
        "interest rate risk reporting depends on Kubernetes batch pipelines",
        "credit model inference latency spikes due to CUDA driver mismatch",
        "FICO monitoring dashboards show latency regressions in us-east",
    ]

    docs_meta: list[dict] = []
    prompts: list[str] = []

    for cluster in CLUSTERS:
        # Decide how many public/internal pairs to create.
        n_pairs = int(round(n * pair_rate))
        n_singles = n - n_pairs

        # Singles
        for i in range(n_singles):
            tenant_id = rng.choice(tenants)
            doc_type = rng.choice(list(cluster.doc_type_weights.keys()), p=np.array(list(cluster.doc_type_weights.values())) / sum(cluster.doc_type_weights.values()))
            owner_team = rng.choice(cluster.owner_teams)
            topic = rng.choice(cluster.topic_hints)
            overlap_hint = rng.choice(overlap_hints) if rng.random() < overlap_rate else None
            prompt = _build_doc_prompt(
                cluster=cluster,
                topic=topic,
                doc_type=doc_type,
                overlap_hint=overlap_hint,
                force_public=(doc_type == "public_summary"),
                force_internal=False,
            )
            doc_id = f"{cluster.name[:3].lower()}_{i:06d}"
            docs_meta.append(
                {
                    "doc_id": doc_id,
                    "cluster": cluster.name,
                    "tenant_id": str(tenant_id),
                    "doc_type": str(doc_type),
                    "owner_team": str(owner_team),
                    "is_pair": False,
                    "pair_id": None,
                }
            )
            prompts.append(prompt)

        # Public/Internal near-duplicate pairs
        for j in range(n_pairs):
            tenant_id = rng.choice([t for t in tenants if t != "global"] or tenants)
            owner_team = rng.choice(cluster.owner_teams)
            topic = rng.choice(cluster.topic_hints)
            overlap_hint = rng.choice(overlap_hints) if rng.random() < overlap_rate else None
            pair_id = f"{cluster.name[:3].lower()}pair_{j:05d}"

            # Public summary doc
            pub_prompt = _build_doc_prompt(
                cluster=cluster,
                topic=topic,
                doc_type="public_summary",
                overlap_hint=overlap_hint,
                force_public=True,
                force_internal=False,
            )
            docs_meta.append(
                {
                    "doc_id": f"{pair_id}_pub",
                    "cluster": cluster.name,
                    "tenant_id": str(tenant_id),
                    "doc_type": "public_summary",
                    "owner_team": str(owner_team),
                    "is_pair": True,
                    "pair_id": pair_id,
                }
            )
            prompts.append(pub_prompt)

            # Internal details doc (near-duplicate but with sensitive lines)
            internal_prompt = _build_doc_prompt(
                cluster=cluster,
                topic=topic,
                doc_type="internal_details",
                overlap_hint=overlap_hint,
                force_public=False,
                force_internal=True,
            )
            docs_meta.append(
                {
                    "doc_id": f"{pair_id}_int",
                    "cluster": cluster.name,
                    "tenant_id": str(tenant_id),
                    "doc_type": "internal_details",
                    "owner_team": str(owner_team),
                    "is_pair": True,
                    "pair_id": pair_id,
                }
            )
            prompts.append(internal_prompt)

    if args.template_only:
        # Map existing 1-line templates into multi-field docs.
        rows: list[dict] = []
        flat_texts: list[tuple[str, str]] = []
        for cluster, texts in zip(CLUSTERS, per_cluster_texts):
            for t in texts:
                flat_texts.append((cluster.name, _clean_line(t)))
        for idx, (cluster_name, line) in enumerate(flat_texts):
            cluster = next(c for c in CLUSTERS if c.name == cluster_name)
            tenant_id = rng.choice(tenants)
            doc_type = rng.choice(list(cluster.doc_type_weights.keys()), p=np.array(list(cluster.doc_type_weights.values())) / sum(cluster.doc_type_weights.values()))
            owner_team = rng.choice(cluster.owner_teams)
            title = line[:72].rstrip(".")
            body = _ensure_sentence_count(line + " " + line + " " + line)
            tags = sorted(list(set(rng.choice(cluster.base_tags, size=min(5, len(cluster.base_tags)), replace=False).tolist())))
            access_level = _infer_access_level(doc_type, tags)
            allowed_roles, allowed_tenants, restricted_tags = _rbac_policy_for_doc(doc_type, tenant_id, tags, access_level)
            created_at = (datetime.now(timezone.utc) - timedelta(days=int(rng.integers(0, 365)))).isoformat()
            body_redacted, redaction_count = _redact_body(body)
            rows.append(
                {
                    "doc_id": f"{cluster_name[:3].lower()}_{idx:06d}",
                    "cluster": cluster_name,
                    "tenant_id": tenant_id,
                    "doc_type": doc_type,
                    "owner_team": owner_team,
                    "created_at": created_at,
                    "title": title,
                    "body": body,
                    "body_redacted": body_redacted,
                    "redaction_count": int(redaction_count),
                    "tags": tags,
                    "access_level": int(access_level),
                    "allowed_roles": allowed_roles,
                    "allowed_tenants": allowed_tenants,
                    "restricted_tags": restricted_tags,
                    "pair_id": None,
                }
            )
    else:
        raw = generate_texts_with_llm(
            model_id=args.gen_model,
            prompts=prompts,
            seed=seed,
            max_new_tokens=max(128, int(args.max_new_tokens)),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            batch_size=int(args.gen_batch_size),
        )

        rows = []
        for meta, out in zip(docs_meta, raw):
            obj = _extract_json_object(out) or {}
            title = _clean_line(str(obj.get("title", "")))[:140] if obj.get("title") else ""
            body = str(obj.get("body", "")).strip() if obj.get("body") else ""
            tags = obj.get("tags") if isinstance(obj.get("tags"), list) else []
            tags = [str(t).strip().lower() for t in tags if str(t).strip()]

            if not title:
                # Fallback: derive title from first sentence.
                first = _sentences(out)
                title = _clean_line(first[0])[:140] if first else f"{meta['cluster']} Document"
            if not body:
                # Fallback: use raw output as body.
                body = out.strip()

            # Enforce 3-10 sentences, and inject controlled overlap if requested but missing.
            body = _ensure_sentence_count(body, 3, 10)

            # Tags: ensure cluster base tags are represented.
            if len(tags) < 3:
                cluster = next(c for c in CLUSTERS if c.name == meta["cluster"])
                extra = rng.choice(cluster.base_tags, size=min(5, len(cluster.base_tags)), replace=False).tolist()
                tags = sorted(list(set(tags + [t.lower() for t in extra])))[:7]

            # For internal details docs, ensure at least one sensitive marker exists.
            if meta["doc_type"] == "internal_details" and not any(x in body for x in ("SECRET:", "PII:", "CREDENTIAL:", "TOKEN:")):
                body = body + "\nSECRET: internal operational details should be redacted for non-admin users."
                tags = sorted(list(set(tags + ["secrets"])))[:7]

            # Create derived fields and RBAC policy.
            access_level = _infer_access_level(meta["doc_type"], tags)
            allowed_roles, allowed_tenants, restricted_tags = _rbac_policy_for_doc(meta["doc_type"], meta["tenant_id"], tags, access_level)
            created_at = (datetime.now(timezone.utc) - timedelta(days=int(rng.integers(0, 365)))).isoformat()
            body_redacted, redaction_count = _redact_body(body)

            rows.append(
                {
                    **meta,
                    "created_at": created_at,
                    "title": title,
                    "body": body,
                    "body_redacted": body_redacted,
                    "redaction_count": int(redaction_count),
                    "tags": tags,
                    "access_level": int(access_level),
                    "allowed_roles": allowed_roles,
                    "allowed_tenants": allowed_tenants,
                    "restricted_tags": restricted_tags,
                }
            )

    df = pd.DataFrame(rows)
    # Add canonical text fields.
    # - text: full-ish content for display/keyword match
    # - embed_text: bounded content for embedding/PCA
    df["text"] = [
        _doc_text(
            title=str(r["title"]),
            body=str(r["body"]),
            tags=list(r.get("tags") or []),
            doc_type=str(r.get("doc_type")),
            tenant_id=str(r.get("tenant_id")),
        )
        for r in rows
    ]
    df["embed_text"] = [
        _embed_text(
            title=str(r["title"]),
            body=str(r.get("body_redacted") or r.get("body") or ""),
            tags=list(r.get("tags") or []),
            doc_type=str(r.get("doc_type")),
            tenant_id=str(r.get("tenant_id")),
        )
        for r in rows
    ]

    # Serialize list fields for CSV friendliness.
    for col in ["tags", "allowed_roles", "allowed_tenants", "restricted_tags"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.dumps(list(x) if isinstance(x, (list, tuple, set)) else [], ensure_ascii=False))

    df = df[df["text"].astype(str).str.len() > 0].reset_index(drop=True)

    # Resolve output paths
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else ""
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, os.path.basename(args.out))
        prefix = os.path.join(out_dir, args.prefix)
    else:
        out_csv = os.path.abspath(args.out)
        prefix = os.path.splitext(out_csv)[0]

    if args.pseudo_embed:
        print(f"✅ Generated {len(df)} rows. Using pseudo-embeddings for x/y/z (demo-only).")
        rng = np.random.default_rng(seed)
        centers = {
            "Finance": (np.array([1.0, 1.0, 0.0]), 0.20),
            "Infrastructure": (np.array([-1.0, 1.0, 0.0]), 0.20),
            "Noise": (np.array([0.0, -1.0, 0.5]), 0.30),
        }
        xyz = np.zeros((len(df), 3), dtype=float)
        for idx, row in df.iterrows():
            center, scale = centers[str(row["cluster"])]
            xyz[idx] = center + rng.normal(0, scale, size=3)
        df[["x", "y", "z"]] = xyz
    else:
        print(f"✅ Generated {len(df)} rows. Embedding on GPU with: {args.embed_model}")
        emb = embed_texts(
            texts=df["embed_text"].tolist(),
            embed_model_id=args.embed_model,
            batch_size=int(args.embed_batch_size),
            seed=seed,
            max_tokens=int(args.embed_max_tokens),
        )

        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        X = scaler.fit_transform(emb)
        pca = PCA(n_components=3, random_state=seed)
        xyz = pca.fit_transform(X)
        df[["x", "y", "z"]] = xyz

        if args.save_embeddings:
            emb_path = f"{prefix}_embeddings.npy"
            pca_path = f"{prefix}_pca.json"
            np.save(emb_path, emb)

            pca_payload = {
                "embed_model": args.embed_model,
                "embed_max_tokens": int(args.embed_max_tokens),
                "scaler": {
                    "mean": scaler.mean_.tolist(),
                    "scale": scaler.scale_.tolist(),
                },
                "pca": {
                    "components": pca.components_.tolist(),
                    "mean": pca.mean_.tolist(),
                    "explained_variance": pca.explained_variance_.tolist(),
                    "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                },
                # Alignment key so notebook can match embeddings rows to docs
                "doc_id_order": df["doc_id"].astype(str).tolist(),
            }
            _write_json(pca_path, pca_payload)
            print(f"✅ Wrote: {emb_path}")
            print(f"✅ Wrote: {pca_path}")

    df.to_csv(out_csv, index=False)
    print(f"✅ Wrote: {out_csv}")
    print("   Columns:", ", ".join(df.columns))

    # Optional: save raw corpus (pre-embedding) artifacts
    if args.save_corpus:
        jsonl_path = f"{prefix}.jsonl"
        txt_path = f"{prefix}.txt"
        meta = {
            "generated_at_unix": int(time.time()),
            "gen_model": None if args.template_only else args.gen_model,
            "embed_model": None if args.pseudo_embed else args.embed_model,
            "n_per_cluster": n,
            "seed": seed,
        }

        with open(jsonl_path, "w", encoding="utf-8") as f:
            # Write a richer record (multi-field + RBAC policy).
            for rec in df.to_dict(orient="records"):
                # Expand JSON-serialized list fields back to lists
                for col in ["tags", "allowed_roles", "allowed_tenants", "restricted_tags"]:
                    if col in rec and isinstance(rec[col], str):
                        try:
                            rec[col] = json.loads(rec[col])
                        except Exception:
                            rec[col] = []
                f.write(json.dumps({**rec, **meta}, ensure_ascii=False) + "\n")

        with open(txt_path, "w", encoding="utf-8") as f:
            for t in df["text"].astype(str).tolist():
                f.write(t.replace("\n", " ").strip() + "\n")

        print(f"✅ Wrote: {jsonl_path}")
        print(f"✅ Wrote: {txt_path}")

    if args.stats:
        sizes_path = f"{prefix}_sizes.txt"
        stats_path = f"{prefix}_stats.json"

        def _maybe_size(p: str) -> int | None:
            try:
                return os.path.getsize(p)
            except OSError:
                return None

        # Estimate tokens using the embedding tokenizer (cheap + always present when embeddings are computed).
        token_estimate = None
        if not args.pseudo_embed:
            try:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(args.embed_model, use_fast=True)
                # Use truncation to avoid max-seq warnings during estimation.
                enc = tok(
                    df["embed_text"].astype(str).tolist(),
                    truncation=True,
                    max_length=int(getattr(tok, "model_max_length", 512) or 512),
                    padding=False,
                    return_attention_mask=False,
                    return_token_type_ids=False,
                )
                token_counts = [len(ids) for ids in enc.get("input_ids", [])]
                token_estimate = {
                    "total_tokens": int(sum(token_counts)),
                    "avg_tokens": float(np.mean(token_counts)),
                    "p95_tokens": float(np.percentile(token_counts, 95)),
                    "max_tokens": int(max(token_counts) if token_counts else 0),
                }
            except Exception:
                token_estimate = None

        per_cluster = (
            df.groupby("cluster")
            .agg(rows=("text", "size"), unique_texts=("text", "nunique"), avg_chars=("text", lambda s: float(np.mean(s.astype(str).str.len()))))
            .reset_index()
            .to_dict(orient="records")
        )

        stats = {
            "generated_at_unix": int(time.time()),
            "template_only": bool(args.template_only),
            "gen_model": None if args.template_only else args.gen_model,
            "embed_model": None if args.pseudo_embed else args.embed_model,
            "seed": seed,
            "n_per_cluster": n,
            "total_rows": int(len(df)),
            "unique_texts": int(df["text"].nunique()),
            "per_cluster": per_cluster,
            "token_estimate": token_estimate,
            "artifacts": {
                "embedded_csv": {"path": out_csv, "bytes": _maybe_size(out_csv)},
                "jsonl": {"path": f"{prefix}.jsonl", "bytes": _maybe_size(f"{prefix}.jsonl")},
                "txt": {"path": f"{prefix}.txt", "bytes": _maybe_size(f"{prefix}.txt")},
            },
        }

        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        lines = [
            f"embedded_csv\t{_maybe_size(out_csv)}\t{out_csv}",
            f"corpus_jsonl\t{_maybe_size(f'{prefix}.jsonl')}\t{prefix}.jsonl",
            f"corpus_txt\t{_maybe_size(f'{prefix}.txt')}\t{prefix}.txt",
            f"stats_json\t{_maybe_size(stats_path)}\t{stats_path}",
        ]
        with open(sizes_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        print(f"✅ Wrote: {stats_path}")
        print(f"✅ Wrote: {sizes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


