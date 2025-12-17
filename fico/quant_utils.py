"""
Quantization Lab Utilities

Helper functions for memory tracking, perplexity calculation, 
benchmarking, and model comparison.
"""

from __future__ import annotations

import gc
import time
import math
import shutil
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Callable
from contextlib import contextmanager
from pathlib import Path

import torch
import numpy as np

# ============================================================================
# CRITICAL FIX: AutoGPTQ Compatibility Patch
# ============================================================================
# Qwen2.5 requires accessing 'attention_type' on layers during the forward pass.
# AutoGPTQ wraps layers in a 'LayerHijacker' which hides this attribute.
# We monkey-patch the hijacker to pass these requests through to the real layer.
try:
    from auto_gptq.quantization.quantizer import LayerHijacker
    
    # Only patch if __getattr__ isn't already defined
    if not hasattr(LayerHijacker, '__getattr__'):
        def getattr_patch(self, name):
            try:
                return getattr(self.module, name)
            except AttributeError:
                raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        LayerHijacker.__getattr__ = getattr_patch
        print("\u2705 AutoGPTQ LayerHijacker patched for Qwen2.5 compatibility.")
except ImportError:
    # If auto_gptq isn't installed, we skip this (handled elsewhere)
    pass
# ============================================================================


# ============================================================================
# Memory Tracking
# ============================================================================

def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# ============================================================================
# Perplexity Calculation
# ============================================================================

def calculate_perplexity_simple(
    model,
    tokenizer,
    text: str,
) -> float:
    """
    Simple perplexity calculation for short texts.
    """
    # Truncate to avoid OOM on very long texts, but keep enough for valid signal
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = encodings.input_ids.to(next(model.parameters()).device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()


# ============================================================================
# Latency Benchmarking
# ============================================================================

@dataclass
class GenerationResult:
    """Result from a generation benchmark."""
    prompt: str
    output: str
    tokens_generated: int
    time_seconds: float
    tokens_per_second: float


def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> GenerationResult:
    """
    Benchmark generation speed for a prompt.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    
    # Warmup
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark runs
    times = []
    final_output = None
    
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append(end - start)
        final_output = outputs
    
    avg_time = sum(times) / len(times)
    output_length = final_output.shape[1]
    tokens_generated = output_length - input_length
    
    output_text = tokenizer.decode(final_output[0][input_length:], skip_special_tokens=True)
    
    return GenerationResult(
        prompt=prompt,
        output=output_text,
        tokens_generated=tokens_generated,
        time_seconds=avg_time,
        tokens_per_second=tokens_generated / avg_time if avg_time > 0 else 0,
    )


# ============================================================================
# Model Loading Logic
# ============================================================================

@dataclass
class ModelLoadResult:
    name: str
    precision: str
    load_time_seconds: float
    memory_mb: float
    model: Any = field(repr=False)
    tokenizer: Any = field(repr=False)


def load_model_base(model_name, precision_name, load_func, **kwargs):
    """Generic loader wrapper to handle timing and memory."""
    clear_gpu_memory()
    start_time = time.perf_counter()
    start_mem = get_gpu_memory_mb()
    
    model, tokenizer = load_func(model_name, **kwargs)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    end_mem = get_gpu_memory_mb()
    
    return ModelLoadResult(
        name=model_name,
        precision=precision_name,
        load_time_seconds=end_time - start_time,
        memory_mb=end_mem - start_mem,
        model=model,
        tokenizer=tokenizer,
    )

def _load_fp16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    return model, tokenizer

def _load_int8(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True, 
        device_map="auto"
    )
    return model, tokenizer

def _load_int4(model_name, quant_type="nf4"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    return model, tokenizer

def _load_gptq(model_name, calibration_text, bits):
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Prepare calibration data
    calibration_examples = [
        tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        for chunk in [calibration_text[i:i+512] for i in range(0, len(calibration_text), 512)]
    ]
    # AutoGPTQ expects a list of dicts with 'input_ids' and 'attention_mask'
    examples = [{"input_ids": c.input_ids, "attention_mask": c.attention_mask} for c in calibration_examples]

    # Quantize
    quantize_config = BaseQuantizeConfig(bits=bits, group_size=128, desc_act=False)
    
    # Load model to CPU/GPU first for quantization
    model = AutoGPTQForCausalLM.from_pretrained(model_name, quantize_config, device_map="auto")
    
    # Run Calibration
    model.quantize(examples)
    
    # Save to temp and reload (crucial for AutoGPTQ to use optimized kernels)
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dir = Path(tmp_dir) / f"gptq_int{bits}"
        model.save_quantized(save_dir, use_safetensors=True)
        tokenizer.save_pretrained(save_dir)
        del model
        clear_gpu_memory()
        
        # Reload the optimized model
        model = AutoGPTQForCausalLM.from_quantized(
            save_dir, 
            device="cuda:0", 
            use_safetensors=True, 
            disable_exllama=False
        )
    return model, tokenizer

# Public Loaders
def load_model_fp16(name): return load_model_base(name, "FP16", _load_fp16)
def load_model_int8(name): return load_model_base(name, "INT8", _load_int8)
def load_model_int4(name, quant_type="nf4"): return load_model_base(name, f"INT4-{quant_type.upper()}", _load_int4, quant_type=quant_type)
def quantize_and_load_gptq(name, text, bits=4): return load_model_base(name, f"GPTQ-INT{bits}", _load_gptq, calibration_text=text, bits=bits)


# ============================================================================
# Benchmarking & Visualization
# ============================================================================

@dataclass
class QuantizationBenchmark:
    precision: str
    bits_per_weight: float
    memory_mb: float
    load_time_s: float
    perplexity: float
    tokens_per_second: float
    
    def to_dict(self) -> dict:
        return asdict(self)

def create_summary_dashboard(benchmarks: list[QuantizationBenchmark]):
    """Create a beautiful Plotly dashboard summary."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import pandas as pd
    
    df = pd.DataFrame([b.to_dict() for b in benchmarks])
    
    # Colors for different precisions
    colors = {
        'FP16': '#636EFA', 
        'INT8': '#EF553B', 
        'INT4-NF4': '#00CC96', 
        'INT4-FP4': '#AB63FA', 
        'GPTQ-INT4': '#FFA15A', 
        'GPTQ-INT2': '#19D3F3'
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "<b>GPU Memory Usage</b> (Lower is Better)",
            "<b>Inference Speed</b> (Higher is Better)",
            "<b>Perplexity / Error</b> (Lower is Better)",
            "<b>Speed vs Quality Tradeoff</b>"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # 1. Memory Usage
    fig.add_trace(
        go.Bar(x=df["precision"], y=df["memory_mb"], marker_color=[colors.get(p, '#888') for p in df["precision"]], text=df["memory_mb"], texttemplate="%{text:.0f} MB", textposition="auto", name="Memory"),
        row=1, col=1
    )
    
    # 2. Inference Speed
    fig.add_trace(
        go.Bar(x=df["precision"], y=df["tokens_per_second"], marker_color=[colors.get(p, '#888') for p in df["precision"]], text=df["tokens_per_second"], texttemplate="%{text:.1f} t/s", textposition="auto", name="Speed"),
        row=1, col=2
    )
    
    # 3. Perplexity (Log Scale handling for INT2)
    fig.add_trace(
        go.Bar(x=df["precision"], y=df["perplexity"], marker_color=[colors.get(p, '#888') for p in df["precision"]], text=df["perplexity"], texttemplate="%{text:.2f}", textposition="outside", name="Perplexity"),
        row=2, col=1
    )
    # If INT2 perplexity is huge (>100), use log scale
    if df["perplexity"].max() > 100:
        fig.update_yaxes(type="log", row=2, col=1, title="Perplexity (Log Scale)")
    else:
        fig.update_yaxes(title="Perplexity", row=2, col=1)

    # 4. Tradeoff Scatter
    fig.add_trace(
        go.Scatter(
            x=df["tokens_per_second"],
            y=df["perplexity"],
            mode="markers+text",
            text=df["precision"],
            textposition="top center",
            marker=dict(size=df["memory_mb"]/50, color=[colors.get(p, '#888') for p in df["precision"]], line=dict(width=2, color='DarkSlateGrey')),
            name="Tradeoff"
        ),
        row=2, col=2
    )
    fig.update_yaxes(title="Perplexity (Lower = Better)", row=2, col=2)
    fig.update_xaxes(title="Tokens / Sec (Higher = Better)", row=2, col=2)

    fig.update_layout(height=900, title_text="<b>Quantization Benchmark: Full Precision vs INT8 vs INT4 vs INT2</b>", showlegend=False, template="plotly_white")
    return fig