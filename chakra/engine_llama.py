"""
Engine: llama.cpp backend for Chakra AI.
Uses llama-cpp-python for SIMD-optimized, multi-threaded inference on CPU.
Expected: 20-50 tokens/sec with Q4_K_M quantized models (~1 GB RAM).
Author: Abhirup Guha (Info Security Solution)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_GGUF_MODEL = "models/chakra_local/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
GGUF_REPO = "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF"
GGUF_FILE = "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"


class LlamaCppBackend:
    """Fast inference using llama.cpp with GGUF quantized models.

    Requirements:
        pip install llama-cpp-python
        GGUF model at models/chakra_local/*.gguf

    Usage:
        backend = LlamaCppBackend()
        if backend.loaded:
            response = backend.generate("Write hello world in Python")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
    ) -> None:
        self.model_path = model_path or self._find_gguf_model()
        self.n_ctx = n_ctx
        self.n_threads = n_threads or (os.cpu_count() or 4)
        self.model_name = "llama.cpp (GGUF)"
        self.loaded = False
        self.model = None

        if self.model_path and Path(self.model_path).exists():
            self._load_model()

    def _find_gguf_model(self) -> Optional[str]:
        """Find any .gguf file in models/chakra_local/"""
        default = Path(DEFAULT_GGUF_MODEL)
        if default.exists():
            return str(default)

        local_dir = Path("models/chakra_local")
        if local_dir.exists():
            gguf_files = list(local_dir.glob("*.gguf"))
            if gguf_files:
                return str(gguf_files[0])

        return None

    def _load_model(self) -> None:
        try:
            from llama_cpp import Llama

            self.model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                verbose=False,
            )
            self.model_name = Path(self.model_path).name
            self.loaded = True
            print(f"[INFO] llama.cpp backend loaded: {self.model_name} ({self.n_threads} threads)")
        except ImportError:
            print("[INFO] llama-cpp-python not installed. Run: pip install llama-cpp-python")
            self.loaded = False
        except Exception as e:
            print(f"[WARN] llama.cpp failed to load model: {e}")
            self.loaded = False

    def generate(
        self,
        prompt: str,
        n_new: int = 192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system: str = "",
    ) -> str:
        """Generate text from prompt using llama.cpp chat completion.

        Args:
            prompt: User message.
            n_new: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            system: Optional system message (never echoed in response).
        """
        if not self.loaded or self.model is None:
            return f"[llama.cpp] Backend not loaded. Prompt: {prompt}"

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=n_new,
                temperature=temperature,
                top_p=top_p,
                stream=False,
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[llama.cpp] Error: {e}"

    def generate_stream(
        self,
        prompt: str,
        n_new: int = 192,
        system: str = "",
    ):
        """Stream tokens as they are generated."""
        if not self.loaded or self.model is None:
            yield f"[llama.cpp] Backend not loaded."
            return

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=n_new,
                temperature=0.7,
                stream=True,
            )
            for chunk in response:
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
        except Exception as e:
            yield f"[llama.cpp] Error: {e}"

    def __call__(self, prompt: str) -> str:
        return self.generate(prompt)

    def check_health(self) -> Dict[str, Any]:
        """Return backend status."""
        return {
            "loaded": self.loaded,
            "model": self.model_name,
            "threads": self.n_threads,
            "context": self.n_ctx,
            "model_path": self.model_path,
        }


def download_gguf_model(force: bool = False) -> Optional[str]:
    """Download Q4_K_M GGUF model from HuggingFace (~1 GB).

    Returns:
        Path to downloaded file, or None if failed.
    """
    output_dir = Path("models/chakra_local")
    output_path = output_dir / GGUF_FILE

    if output_path.exists() and not force:
        print(f"[INFO] GGUF model already exists: {output_path}")
        return str(output_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        print(f"[INFO] Downloading {GGUF_FILE} from {GGUF_REPO}...")
        path = hf_hub_download(
            repo_id=GGUF_REPO,
            filename=GGUF_FILE,
            local_dir=str(output_dir),
        )
        print(f"[SUCCESS] GGUF model downloaded to: {path}")
        return path
    except ImportError:
        print("[ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        return None
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        return None


if __name__ == "__main__":
    import sys
    if "--download" in sys.argv:
        download_gguf_model()
    else:
        backend = LlamaCppBackend()
        if backend.loaded:
            health = backend.check_health()
            print(f"Backend: {health}")
            print(f"\nTest prompt: 'Write hello world in Python'")
            print(backend.generate("Write hello world in Python", n_new=64))
        else:
            print("Backend not loaded. Install llama-cpp-python and download GGUF model.")
            print("  pip install llama-cpp-python")
            print("  python chakra/engine_llama.py --download")
