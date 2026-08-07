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
        enable_turbo: bool = True,
        draft_model_path: Optional[str] = None,
    ) -> None:
        self.model_path = model_path or self._find_gguf_model()
        self.n_ctx = n_ctx
        self.enable_turbo = enable_turbo
        self.draft_model_path = draft_model_path

        # Determine physical CPU cores (avoids hyperthread cache thrashing on CPU GEMM)
        phys_cores = os.cpu_count() or 4
        try:
            import psutil
            phys_cores = psutil.cpu_count(logical=False) or phys_cores
        except ImportError:
            pass

        self.n_threads = n_threads or phys_cores
        self.model_name = "llama.cpp (GGUF Turbo)"
        self.loaded = False
        self.model = None
        self.cache = None

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
            from llama_cpp import Llama, LlamaRAMCache

            # Configure high-performance execution parameters
            kwargs: Dict[str, Any] = {
                "model_path": self.model_path,
                "n_ctx": self.n_ctx,
                "n_threads": self.n_threads,
                "n_threads_batch": os.cpu_count(),
                "n_batch": 2048,
                "n_ubatch": 512,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "verbose": False,
            }

            if self.enable_turbo:
                try:
                    kwargs["flash_attn"] = True
                except Exception:
                    pass

            # Auto-detect draft model for speculative decoding if available
            if not self.draft_model_path:
                local_dir = Path("models/chakra_local")
                if local_dir.exists():
                    draft_files = list(local_dir.glob("*0.5B*.gguf")) or list(local_dir.glob("*draft*.gguf"))
                    if draft_files and str(draft_files[0]) != self.model_path:
                        self.draft_model_path = str(draft_files[0])

            if self.draft_model_path and Path(self.draft_model_path).exists():
                try:
                    from llama_cpp import LlamaDraftModel
                    kwargs["draft_model"] = LlamaDraftModel(model_path=self.draft_model_path)
                    print(f"[INFO] Speculative Draft Decoding active: {Path(self.draft_model_path).name}")
                except Exception:
                    pass

            self.model = Llama(**kwargs)

            # Enable stateful RAM KV cache (eliminates prompt re-encoding across turns)
            try:
                self.cache = LlamaRAMCache(capacity_bytes=512 * 1024 * 1024)
                self.model.set_cache(self.cache)
            except Exception:
                pass

            self.model_name = Path(self.model_path).name
            self.loaded = True
            self._grammar_cache = None
            print(f"[INFO] llama.cpp Turbo Backend loaded: {self.model_name} ({self.n_threads} phys threads, flash_attn=True, KV cache=active)")
        except ImportError:
            print("[INFO] llama-cpp-python not installed. Run: pip install llama-cpp-python")
            self.loaded = False
        except Exception as e:
            # Fallback without flash_attn if unsupported
            try:
                from llama_cpp import Llama
                self.model = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, verbose=False)
                self.model_name = Path(self.model_path).name
                self.loaded = True
                self._grammar_cache = None
                print(f"[INFO] llama.cpp Standard Backend loaded: {self.model_name}")
            except Exception as ex:
                print(f"[WARN] llama.cpp failed to load model: {ex}")
                self.loaded = False

    def _build_messages(
        self, prompt: Optional[str], system: str, messages: Optional[list]
    ) -> list:
        """Returns a chat messages list: `messages` verbatim if given (multi-turn conversation),
        otherwise a fresh single-turn [system?, user] list built from `prompt`/`system`."""
        if messages is not None:
            return messages
        built = []
        if system:
            built.append({"role": "system", "content": system})
        built.append({"role": "user", "content": prompt or ""})
        return built

    def _chat(
        self,
        messages: list,
        max_tokens: int,
        temperature: float = 0.0,
        top_p: float = 0.9,
        grammar: Any = None,
        stop: Optional[list] = None,
    ) -> str:
        """Shared low-level chat completion call. Raises on failure (callers handle fallback)."""
        kwargs: Dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            kwargs["stop"] = stop
        if grammar is not None:
            kwargs["grammar"] = grammar
        response = self.model.create_chat_completion(**kwargs)
        return response["choices"][0]["message"]["content"]

    def generate_choice(
        self,
        prompt: Optional[str] = None,
        choices: Optional[list] = None,
        system: str = "",
        messages: Optional[list] = None,
    ) -> str:
        """Generate exactly one of a fixed set of literal string choices via grammar-constrained
        decoding. Used for the tool-loop's CALL_TOOL/FINAL_ANSWER decision step: near-zero token
        cost, and the grammar makes an invalid/ambiguous answer impossible.

        Args:
            prompt: User message (ignored if `messages` is given).
            choices: List of literal strings the model must pick from.
            system: Optional system message (ignored if `messages` is given).
            messages: Optional full multi-turn chat messages list, for callers maintaining a
                growing conversation (preferred over `prompt` — lets llama.cpp reuse the shared
                prefix across turns instead of re-encoding a flattened string each call).

        Returns:
            One of `choices` (the first choice if the backend isn't loaded or grammar fails).
        """
        choices = choices or []
        if not choices:
            return ""
        if not self.loaded or self.model is None:
            return choices[0]

        try:
            from llama_cpp import LlamaGrammar

            cache_key = tuple(choices)
            if not hasattr(self, "_choice_grammar_cache"):
                self._choice_grammar_cache = {}
            grammar = self._choice_grammar_cache.get(cache_key)
            if grammar is None:
                # Longest-first so the grammar doesn't stop early on a shorter choice
                # that happens to be a prefix of a longer one.
                sorted_choices = sorted(choices, key=len, reverse=True)
                alternatives = " | ".join('"' + c.replace('"', '\\"') + '"' for c in sorted_choices)
                grammar = LlamaGrammar.from_string(f"root ::= ({alternatives})")
                self._choice_grammar_cache[cache_key] = grammar

            msgs = self._build_messages(prompt, system, messages)
            text = self._chat(msgs, max_tokens=16, temperature=0.0, grammar=grammar).strip()
            return text if text in choices else choices[0]
        except Exception:
            return choices[0]

    def generate_json(
        self,
        prompt: Optional[str] = None,
        schema: Optional[Dict[str, Any]] = None,
        system: str = "",
        n_new: int = 256,
        messages: Optional[list] = None,
    ) -> Optional[Dict[str, Any]]:
        """Generate JSON constrained to exactly match a JSON schema via
        `LlamaGrammar.from_json_schema`. Guarantees parseable, schema-valid output from even a
        small model — used for tool-call argument generation.

        Args:
            prompt: User message (ignored if `messages` is given).
            schema: JSON schema dict the output must conform to.
            system: Optional system message (ignored if `messages` is given).
            n_new: Max tokens to generate.
            messages: Optional full multi-turn chat messages list (see `generate_choice`).

        Returns:
            Parsed dict matching schema, or None if the backend isn't loaded or generation/parse
            fails (caller should treat None as "tool call failed, fall back").
        """
        if not self.loaded or self.model is None or not schema:
            return None

        try:
            import json as _json

            from llama_cpp import LlamaGrammar

            if not hasattr(self, "_json_grammar_cache"):
                self._json_grammar_cache = {}
            cache_key = _json.dumps(schema, sort_keys=True)
            grammar = self._json_grammar_cache.get(cache_key)
            if grammar is None:
                grammar = LlamaGrammar.from_json_schema(cache_key)
                self._json_grammar_cache[cache_key] = grammar

            msgs = self._build_messages(prompt, system, messages)
            text = self._chat(msgs, max_tokens=n_new, temperature=0.0, grammar=grammar)
            return _json.loads(text)
        except Exception:
            return None

    def get_python_grammar(self) -> Any:
        """
        Returns LlamaGrammar instance for constraining output tokens to Python code.
        """
        if getattr(self, "_grammar_cache", None) is not None:
            return self._grammar_cache
        try:
            from llama_cpp import LlamaGrammar
            grammar_str = 'root ::= (code-line "\\n")*\ncode-line ::= [^\\n]*'
            self._grammar_cache = LlamaGrammar.from_string(grammar_str)
            return self._grammar_cache
        except Exception:
            return None

    def generate(
        self,
        prompt: Optional[str] = None,
        n_new: int = 192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        system: str = "",
        use_grammar: bool = False,
        messages: Optional[list] = None,
    ) -> str:
        """Generate text from prompt using llama.cpp chat completion.

        Args:
            prompt: User message (ignored if `messages` is given).
            n_new: Max tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling.
            system: Optional system message (never echoed in response; ignored if `messages`
                is given).
            use_grammar: Optional boolean to enable grammar-constrained token sampling.
            messages: Optional full multi-turn chat messages list (see `generate_choice`).
        """
        if not self.loaded or self.model is None:
            return f"[llama.cpp] Backend not loaded. Prompt: {prompt}"

        try:
            msgs = self._build_messages(prompt, system, messages)
            grammar = self.get_python_grammar() if use_grammar else None
            return self._chat(
                msgs,
                max_tokens=n_new,
                temperature=temperature,
                top_p=top_p,
                grammar=grammar,
                stop=["<|im_end|>", "<|endoftext|>"],
            )
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
            yield "[llama.cpp] Backend not loaded."
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
            print("\nTest prompt: 'Write hello world in Python'")
            print(backend.generate("Write hello world in Python", n_new=64))
        else:
            print("Backend not loaded. Install llama-cpp-python and download GGUF model.")
            print("  pip install llama-cpp-python")
            print("  python chakra/engine_llama.py --download")
