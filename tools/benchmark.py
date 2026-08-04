"""
One-time system benchmark for Chakra-AI.
Measures actual tokens/sec on this hardware and saves optimal settings.
Runs once per system, caches results in .chakra_benchmark.json.
Author & Creator: Abhirup Guha (Info Security Solution)
"""
import json
import sys
import time
from pathlib import Path

BENCHMARK_FILE = Path(".chakra_benchmark.json")
TARGET_SECONDS = 45  # Target generation time in seconds (70% of a reasonable wait)
CPU_USAGE_FACTOR = 0.7  # Use 70% of measured capacity


def run_benchmark() -> dict:
    """Run the benchmark and return results."""
    print("=" * 60, flush=True)
    print("  Chakra-AI System Benchmark", flush=True)
    print("  Measuring optimal tokens/sec for your hardware...", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    # Step 1: Import and load model
    print("[1/3] Loading model...", flush=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_path = "models/chakra_local"
    if not Path(model_path).exists():
        print("[ERROR] Model not found at models/chakra_local/", flush=True)
        print("        Run: python tools/download_model.py", flush=True)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    ).to("cpu")
    model.eval()

    load_time = time.time() - t0
    print(f"       Model loaded in {load_time:.1f}s", flush=True)

    # Step 2: Benchmark generation speed
    print("[2/3] Benchmarking generation speed...", flush=True)

    # Warmup run (discard)
    inputs = tok("Write hello world in Python:", return_tensors="pt", return_attention_mask=True)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=8, pad_token_id=tok.eos_token_id)

    # Actual benchmark: generate 64 tokens
    bench_tokens = 64
    inputs = tok("Write a simple Python function that adds two numbers:", return_tensors="pt", return_attention_mask=True)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=bench_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.time() - t0

    actual_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_sec = actual_tokens / elapsed if elapsed > 0 else 1.0

    print(f"       Generated {actual_tokens} tokens in {elapsed:.1f}s", flush=True)
    print(f"       Speed: {tokens_per_sec:.2f} tokens/sec", flush=True)

    # Step 3: Calculate optimal settings
    print("[3/3] Calculating optimal settings...", flush=True)

    # Target: generate for TARGET_SECONDS at 70% CPU usage
    optimal_tokens = int(tokens_per_sec * TARGET_SECONDS * CPU_USAGE_FACTOR)
    optimal_tokens = max(64, min(optimal_tokens, 1024))  # Clamp between 64-1024

    # Estimate time for different token counts
    estimates = {
        "64_tokens": f"{64 / tokens_per_sec:.0f}s",
        "128_tokens": f"{128 / tokens_per_sec:.0f}s",
        "192_tokens": f"{192 / tokens_per_sec:.0f}s",
        "256_tokens": f"{256 / tokens_per_sec:.0f}s",
        f"{optimal_tokens}_optimal": f"{optimal_tokens / tokens_per_sec:.0f}s",
    }

    results = {
        "tokens_per_sec": round(tokens_per_sec, 2),
        "optimal_gen_tokens": optimal_tokens,
        "target_seconds": TARGET_SECONDS,
        "cpu_usage_factor": CPU_USAGE_FACTOR,
        "model_load_time": round(load_time, 1),
        "bench_tokens_generated": actual_tokens,
        "bench_time": round(elapsed, 2),
        "estimates": estimates,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    BENCHMARK_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(flush=True)
    print("=" * 60, flush=True)
    print("  Benchmark Complete!", flush=True)
    print(f"  Speed: {tokens_per_sec:.2f} tokens/sec", flush=True)
    print(f"  Optimal gen_tokens: {optimal_tokens}", flush=True)
    print(f"  Est. generation time: {optimal_tokens / tokens_per_sec:.0f}s", flush=True)
    print(f"  Results saved to: {BENCHMARK_FILE}", flush=True)
    print("=" * 60, flush=True)

    # Cleanup
    del model
    del tok
    import gc
    gc.collect()

    return results


def load_benchmark() -> dict:
    """Load cached benchmark results if available."""
    if BENCHMARK_FILE.exists():
        try:
            return json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_optimal_gen_tokens(default: int = 192) -> int:
    """Get optimal gen_tokens from benchmark, or default if not benchmarked."""
    results = load_benchmark()
    return results.get("optimal_gen_tokens", default)


def get_tokens_per_sec(default: float = 4.0) -> float:
    """Get measured tokens/sec from benchmark, or default if not benchmarked."""
    results = load_benchmark()
    return results.get("tokens_per_sec", default)


if __name__ == "__main__":
    run_benchmark()
