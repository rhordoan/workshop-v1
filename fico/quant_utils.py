"""
Quantization Lab Utilities

Helper functions for memory tracking, perplexity calculation, 
benchmarking, and model comparison.
"""

from __future__ import annotations

import gc
import time
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable
from contextlib import contextmanager

import torch
import numpy as np


# ============================================================================
# Memory Tracking
# ============================================================================

def get_gpu_memory_mb() -> float:
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def get_gpu_memory_reserved_mb() -> float:
    """Get total GPU memory reserved in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / (1024 * 1024)
    return 0.0


def get_gpu_memory_peak_mb() -> float:
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_model_size_mb(model) -> float:
    """Calculate model size in MB based on parameters."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 * 1024)


def count_parameters(model) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total,
        "total_millions": total / 1e6,
        "total_billions": total / 1e9,
        "trainable": trainable,
        "frozen": total - trainable,
    }


@contextmanager
def track_memory():
    """Context manager to track GPU memory usage."""
    clear_gpu_memory()
    start_mem = get_gpu_memory_mb()
    
    yield
    
    end_mem = get_gpu_memory_mb()
    peak_mem = get_gpu_memory_peak_mb()
    
    print(f"Memory: {start_mem:.1f} MB -> {end_mem:.1f} MB (peak: {peak_mem:.1f} MB)")
    print(f"Delta: +{end_mem - start_mem:.1f} MB")


# ============================================================================
# Perplexity Calculation
# ============================================================================

def calculate_perplexity(
    model,
    tokenizer,
    text: str,
    max_length: int = 512,
    stride: int = 256,
) -> float:
    """
    Calculate perplexity of a model on given text.
    
    Lower perplexity = model is more confident/accurate.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to evaluate
        max_length: Maximum sequence length
        stride: Stride for sliding window
    
    Returns:
        Perplexity score (float)
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    
    device = next(model.parameters()).device
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # Mask prefix tokens
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


def calculate_perplexity_simple(
    model,
    tokenizer,
    text: str,
) -> float:
    """
    Simple perplexity calculation for short texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer  
        text: Text to evaluate
    
    Returns:
        Perplexity score (float)
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
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
    first_token_time: float = 0.0


def benchmark_generation(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    num_runs: int = 3,
    warmup_runs: int = 1,
    temperature: float = 0.7,
    do_sample: bool = True,
) -> GenerationResult:
    """
    Benchmark generation speed for a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        num_runs: Number of benchmark runs (averaged)
        warmup_runs: Warmup runs (not counted)
        temperature: Sampling temperature
        do_sample: Whether to sample (vs greedy)
    
    Returns:
        GenerationResult with timing info
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
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    
    # Synchronize before timing
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
                temperature=temperature,
                do_sample=do_sample,
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
    
    output_text = tokenizer.decode(
        final_output[0][input_length:],
        skip_special_tokens=True
    )
    
    return GenerationResult(
        prompt=prompt,
        output=output_text,
        tokens_generated=tokens_generated,
        time_seconds=avg_time,
        tokens_per_second=tokens_generated / avg_time if avg_time > 0 else 0,
    )


def benchmark_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 50,
    num_runs: int = 2,
) -> list[GenerationResult]:
    """Benchmark multiple prompts."""
    results = []
    for prompt in prompts:
        result = benchmark_generation(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            num_runs=num_runs,
        )
        results.append(result)
    return results


# ============================================================================
# Model Loading with Timing
# ============================================================================

@dataclass
class ModelLoadResult:
    """Result from loading a model."""
    name: str
    precision: str
    load_time_seconds: float
    memory_mb: float
    peak_memory_mb: float
    model: Any = field(repr=False)
    tokenizer: Any = field(repr=False)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "precision": self.precision,
            "load_time_seconds": self.load_time_seconds,
            "memory_mb": self.memory_mb,
            "peak_memory_mb": self.peak_memory_mb,
        }


def load_model_fp16(model_name: str, trust_remote_code: bool = True) -> ModelLoadResult:
    """Load model in FP16 precision."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    clear_gpu_memory()
    
    start_time = time.perf_counter()
    start_mem = get_gpu_memory_mb()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    end_mem = get_gpu_memory_mb()
    peak_mem = get_gpu_memory_peak_mb()
    
    return ModelLoadResult(
        name=model_name,
        precision="FP16",
        load_time_seconds=end_time - start_time,
        memory_mb=end_mem - start_mem,
        peak_memory_mb=peak_mem,
        model=model,
        tokenizer=tokenizer,
    )


def load_model_int8(model_name: str, trust_remote_code: bool = True) -> ModelLoadResult:
    """Load model in INT8 precision using bitsandbytes."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    clear_gpu_memory()
    
    start_time = time.perf_counter()
    start_mem = get_gpu_memory_mb()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    end_mem = get_gpu_memory_mb()
    peak_mem = get_gpu_memory_peak_mb()
    
    return ModelLoadResult(
        name=model_name,
        precision="INT8",
        load_time_seconds=end_time - start_time,
        memory_mb=end_mem - start_mem,
        peak_memory_mb=peak_mem,
        model=model,
        tokenizer=tokenizer,
    )


def load_model_int4(
    model_name: str,
    quant_type: str = "nf4",
    trust_remote_code: bool = True,
) -> ModelLoadResult:
    """
    Load model in INT4 precision using bitsandbytes.
    
    Args:
        model_name: HuggingFace model name
        quant_type: "nf4" (normalized float) or "fp4"
        trust_remote_code: Trust remote code for model loading
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    clear_gpu_memory()
    
    start_time = time.perf_counter()
    start_mem = get_gpu_memory_mb()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Nested quantization
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    end_mem = get_gpu_memory_mb()
    peak_mem = get_gpu_memory_peak_mb()
    
    return ModelLoadResult(
        name=model_name,
        precision=f"INT4-{quant_type.upper()}",
        load_time_seconds=end_time - start_time,
        memory_mb=end_mem - start_mem,
        peak_memory_mb=peak_mem,
        model=model,
        tokenizer=tokenizer,
    )


# ============================================================================
# Comparison & Visualization Helpers
# ============================================================================

@dataclass
class QuantizationBenchmark:
    """Complete benchmark results for a quantization level."""
    precision: str
    bits_per_weight: float
    memory_mb: float
    load_time_s: float
    perplexity: float
    tokens_per_second: float
    sample_outputs: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


def compare_outputs(
    models: dict[str, tuple],  # {precision: (model, tokenizer)}
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
) -> dict[str, str]:
    """
    Generate outputs from multiple models for comparison.
    
    Args:
        models: Dict mapping precision name to (model, tokenizer) tuple
        prompt: Input prompt
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Dict mapping precision name to generated output
    """
    outputs = {}
    
    for precision, (model, tokenizer) in models.items():
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        
        output_text = tokenizer.decode(
            generated[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        outputs[precision] = output_text
    
    return outputs


def format_comparison_table(benchmarks: list[QuantizationBenchmark]) -> str:
    """Format benchmark results as a nice table."""
    header = (
        "┌────────────┬───────┬──────────┬──────────┬────────────┬───────────┐\n"
        "│ Precision  │ Bits  │ Memory   │ Load(s)  │ Perplexity │ Tok/s     │\n"
        "├────────────┼───────┼──────────┼──────────┼────────────┼───────────┤"
    )
    
    rows = []
    for b in benchmarks:
        row = (
            f"│ {b.precision:<10} │ {b.bits_per_weight:>5.1f} │ "
            f"{b.memory_mb:>7.1f}MB │ {b.load_time_s:>7.2f}s │ "
            f"{b.perplexity:>10.2f} │ {b.tokens_per_second:>8.1f} │"
        )
        rows.append(row)
    
    footer = "└────────────┴───────┴──────────┴──────────┴────────────┴───────────┘"
    
    return header + "\n" + "\n".join(rows) + "\n" + footer


# ============================================================================
# Visualization Functions (for Plotly)
# ============================================================================

def create_memory_chart(benchmarks: list[QuantizationBenchmark]):
    """Create a bar chart comparing memory usage."""
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame([b.to_dict() for b in benchmarks])
    
    fig = px.bar(
        df,
        x="precision",
        y="memory_mb",
        title="GPU Memory Usage by Quantization Level",
        labels={"memory_mb": "Memory (MB)", "precision": "Precision"},
        text="memory_mb",
    )
    fig.update_traces(texttemplate="%{text:.0f} MB", textposition="outside")
    fig.update_layout(showlegend=False)
    
    return fig


def create_speed_chart(benchmarks: list[QuantizationBenchmark]):
    """Create a bar chart comparing generation speed."""
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame([b.to_dict() for b in benchmarks])
    
    fig = px.bar(
        df,
        x="precision",
        y="tokens_per_second",
        title="Generation Speed by Quantization Level",
        labels={"tokens_per_second": "Tokens/Second", "precision": "Precision"},
        text="tokens_per_second",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(showlegend=False)
    
    return fig


def create_perplexity_chart(benchmarks: list[QuantizationBenchmark]):
    """Create a line chart showing perplexity vs bits per weight."""
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame([b.to_dict() for b in benchmarks])
    df = df.sort_values("bits_per_weight", ascending=False)
    
    fig = px.line(
        df,
        x="bits_per_weight",
        y="perplexity",
        title="Perplexity vs Bits per Weight (Lower is Better)",
        labels={"bits_per_weight": "Bits per Weight", "perplexity": "Perplexity"},
        markers=True,
    )
    fig.update_xaxes(autorange="reversed")  # Higher bits on left
    
    return fig


def create_tradeoff_chart(benchmarks: list[QuantizationBenchmark]):
    """Create a scatter plot showing speed vs quality tradeoff."""
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame([b.to_dict() for b in benchmarks])
    
    fig = px.scatter(
        df,
        x="tokens_per_second",
        y="perplexity",
        size="memory_mb",
        color="precision",
        title="Speed vs Quality Tradeoff (bubble size = memory)",
        labels={
            "tokens_per_second": "Speed (tokens/sec)",
            "perplexity": "Perplexity (lower is better)",
        },
        hover_data=["memory_mb"],
    )
    
    # Add annotations
    for _, row in df.iterrows():
        fig.add_annotation(
            x=row["tokens_per_second"],
            y=row["perplexity"],
            text=row["precision"],
            showarrow=False,
            yshift=20,
        )
    
    return fig


def create_summary_dashboard(benchmarks: list[QuantizationBenchmark]):
    """Create a summary dashboard with all charts."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import pandas as pd
    
    df = pd.DataFrame([b.to_dict() for b in benchmarks])
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Memory Usage (MB)",
            "Generation Speed (tok/s)",
            "Perplexity (lower is better)",
            "Speed vs Quality Tradeoff"
        ),
    )
    
    # Memory bar
    fig.add_trace(
        go.Bar(x=df["precision"], y=df["memory_mb"], name="Memory"),
        row=1, col=1
    )
    
    # Speed bar
    fig.add_trace(
        go.Bar(x=df["precision"], y=df["tokens_per_second"], name="Speed"),
        row=1, col=2
    )
    
    # Perplexity bar
    fig.add_trace(
        go.Bar(x=df["precision"], y=df["perplexity"], name="Perplexity"),
        row=2, col=1
    )
    
    # Tradeoff scatter
    fig.add_trace(
        go.Scatter(
            x=df["tokens_per_second"],
            y=df["perplexity"],
            mode="markers+text",
            text=df["precision"],
            textposition="top center",
            marker=dict(size=df["memory_mb"] / 50),
            name="Tradeoff"
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Quantization Benchmark Summary",
        showlegend=False,
    )
    
    return fig

