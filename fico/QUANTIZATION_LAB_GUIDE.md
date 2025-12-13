# Quantization Lab - Quick Start Guide

## What You'll Learn

In this lab, you'll explore model quantization by:
- Loading Qwen2.5-1.5B at different precision levels (FP16, INT8, INT4)
- Measuring memory usage, inference speed, and quality
- Comparing outputs side-by-side
- Finding optimal tradeoffs for your use case

## Prerequisites

### Required
- GPU with CUDA support (tested on A100)
- At least 4GB VRAM
- Python 3.10+

### Packages
Already installed in your `.venv`:
- ✅ PyTorch 2.9.1+cu128
- ✅ bitsandbytes 0.49.0
- ✅ transformers 4.44+
- ✅ accelerate 0.33+
- ✅ auto-gptq 0.7.1
- ✅ optimum 2.0.0

## Quick Start

### Step 1: Restart Kernel
After installing packages, restart your Jupyter kernel:
- **Kernel → Restart Kernel** (or press `00`)

### Step 2: Open Notebook
```bash
cd /home/shadeform/workshop-v1/fico
jupyter lab day3_01_quantization_lab.ipynb
```

### Step 3: Run Cells
Run cells sequentially. The lab will:
1. Load models (takes 2-3 min per precision level)
2. Run benchmarks
3. Show interactive comparisons

## Expected Results

### Memory Usage (Qwen2.5-1.5B)
| Precision | Memory | Compression |
|-----------|--------|-------------|
| FP16 | ~3.0 GB | Baseline |
| INT8 | ~1.5 GB | 2x smaller |
| INT4-NF4 | ~0.8 GB | 4x smaller |

### Quality vs Speed
- **FP16**: Best quality, baseline speed
- **INT8**: 95-98% quality retained, similar or faster speed
- **INT4**: 85-95% quality retained, often faster

## Troubleshooting

### "bitsandbytes not found" after restart
```bash
cd /home/shadeform/workshop-v1/fico
./scripts/install_bitsandbytes.sh
# Then restart kernel again
```

### Out of memory errors
- Run cells one at a time (don't keep all models loaded)
- Reduce `max_new_tokens` in generation
- Skip FP16 if you have limited VRAM

### Model download is slow
First load will download ~3GB model from HuggingFace.
Subsequent runs use cached model.

### Kernel keeps restarting
- Your GPU might not have enough VRAM
- Try closing other notebooks
- Skip FP16 benchmark

## Workshop Flow (60 min)

| Time | Activity | Cells |
|------|----------|-------|
| 0-5 min | Setup, verify installation | 1-4 |
| 5-10 min | Learn quantization theory | 5-6 |
| 10-25 min | Load models, see memory usage | 7-12 |
| 25-40 min | Run benchmarks, see charts | 13-18 |
| 40-55 min | Quality arena, voting | 19-21 |
| 55-60 min | Exercises, wrap-up | 22-27 |

## Files

| File | Purpose |
|------|---------|
| `day3_01_quantization_lab.ipynb` | Main notebook |
| `quant_utils.py` | Helper functions |
| `calibration_texts/fico_calibration.txt` | Text for perplexity |
| `scripts/install_bitsandbytes.sh` | Installation script |

## Key Metrics Explained

### Perplexity
- Measures how "surprised" the model is by text
- Lower = better (model is more confident)
- Calculated as: `exp(cross_entropy_loss)`
- Good baseline: < 20 for general text

### Tokens per Second
- How fast the model generates
- Higher = faster
- Depends on: model size, precision, GPU, batch size

### Memory
- GPU VRAM used by model weights
- Formula: `(params × bits) / (8 × 10^9)` GB
- Doesn't include activation memory during inference

## Pro Tips

1. **Load one at a time**: Prevents OOM errors, gives accurate memory measurements
2. **Use warmup runs**: First generation is slower (CUDA initialization)
3. **Clear memory**: Always `clear_gpu_memory()` between models
4. **Compare on same prompts**: Ensures fair quality comparison
5. **Consider task type**: Math needs precision, creative text is more forgiving

## Next Steps

After this lab, explore:
- **GPTQ** for production 4-bit models
- **AWQ** for better 4-bit quality
- **GGUF/llama.cpp** for CPU inference
- **Speculative decoding** for faster generation
- **Mixed precision** strategies

