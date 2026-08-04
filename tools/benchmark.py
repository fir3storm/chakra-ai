"""
One-time system benchmark for Chakra-AI.
Tries llama.cpp first (fastest), falls back to PyTorch.
Author & Creator: Abhirup Guha (Info Security Solution)
"""
import json
import sys
import time
from pathlib import Path

BENCHMARK_FILE = Path(".chakra_benchmark.json")
TARGET_SECONDS = 15  # Under 15s for code gen with fast backend
CPU_USAGE_FACTOR = 1.0


def run_benchmark() -> dict:
    """Run the benchmark and return results."""
    print("=" * 60, flush=True)
    print("  Chakra-AI System Benchmark", flush=True)
    print("  Finding fastest backend on your hardware...", flush=True)
    print("=" * 60, flush=True)
    print(flush=True)

    tokens_per_sec = 0
    backend_name = "unknown"

    # Step 1: Try llama.cpp GGUF first (fastest)
    print("[1/2] Checking for fast llama.cpp backend...", flush=True)
    try:
        from chakra.engine_llama import LlamaCppBackend
        backend = LlamaCppBackend()
        if backend.loaded:
            backend_name = backend.model_name
            backend.generate("Hello", n_new=4)  # warmup

            t0 = time.time()
            backend.generate("Write a simple Python function that adds two numbers:", n_new=64)
            elapsed = time.time() - t0
            tokens_per_sec = 64 / elapsed if elapsed > 0 else 1.0
            print(f"       llama.cpp ({backend_name}): {tokens_per_sec:.1f} tokens/sec ({elapsed:.1f}s)", flush=True)
    except Exception as e:
        print(f"       llama.cpp not available: {e}", flush=True)

    # Step 2: PyTorch fallback
    if tokens_per_sec < 1:
        print("[2/2] Trying PyTorch backend...", flush=True)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_path = "models/chakra_local"
            tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True,
                torch_dtype=torch.float16, low_cpu_mem_usage=True,
            ).to("cpu")
            model.eval()

            inputs = tok("Hello:", return_tensors="pt")
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=4, pad_token_id=tok.eos_token_id)

            inputs = tok("Write a simple Python function that adds two numbers:", return_tensors="pt")
            t0 = time.time()
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7, pad_token_id=tok.eos_token_id)
            elapsed = time.time() - t0
            tokens_per_sec = 64 / elapsed if elapsed > 0 else 1.0
            backend_name = "PyTorch float16"
            print(f"       PyTorch (float16): {tokens_per_sec:.1f} tokens/sec ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            print(f"[ERROR] PyTorch failed: {e}", flush=True)
            sys.exit(1)

    # Step 3: Calculate optimal settings
    print("[2/2] Calculating optimal settings...", flush=True)

    optimal_tokens = int(tokens_per_sec * TARGET_SECONDS * CPU_USAGE_FACTOR)
    optimal_tokens = max(64, min(optimal_tokens, 2048))

    results = {
        "tokens_per_sec": round(tokens_per_sec, 2),
        "backend": backend_name,
        "optimal_gen_tokens": optimal_tokens,
        "target_seconds": TARGET_SECONDS,
        "estimates": {
            "64_tokens":  f"{64 / max(tokens_per_sec, 0.01):.1f}s",
            "256_tokens": f"{256 / max(tokens_per_sec, 0.01):.1f}s",
            "512_tokens": f"{512 / max(tokens_per_sec, 0.01):.1f}s",
            f"{optimal_tokens}_optimal": f"{optimal_tokens / max(tokens_per_sec, 0.01):.1f}s",
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    BENCHMARK_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"  Backend: {backend_name}", flush=True)
    print(f"  Speed: {tokens_per_sec:.1f} tokens/sec", flush=True)
    print(f"  Optimal gen_tokens: {optimal_tokens}", flush=True)
    print(f"  Gen time: ~{optimal_tokens / max(tokens_per_sec, 0.01):.0f}s", flush=True)
    print(f"  Results saved to: {BENCHMARK_FILE}", flush=True)
    print("=" * 60, flush=True)

    import gc; gc.collect()
    return results


def load_benchmark() -> dict:
    if BENCHMARK_FILE.exists():
        try:
            return json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def get_optimal_gen_tokens(default: int = 512) -> int:
    results = load_benchmark()
    return results.get("optimal_gen_tokens", default)


def get_tokens_per_sec(default: float = 300) -> float:
    results = load_benchmark()
    return results.get("tokens_per_sec", default)


if __name__ == "__main__":
    run_benchmark()
