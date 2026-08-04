"""
Engine A: kimi-k3-in-c subprocess backend for full Kimi K3 inference.
Wraps the k3 C binary for bit-exact, 8.24 GB peak RSS inference on Linux.
Author & Creator: Abhirup Guha (Info Security Solution)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class KimiCBackend:
    """
    Wraps the kimi-k3-in-c binary for subprocess-based Kimi K3 inference.
    
    Requires:
        - k3 binary (built from https://github.com/FareedKhan-dev/kimi-k3-in-c)
        - 1.56 TB Kimi K3 checkpoint on disk
        - Packed trunk directory (from pack-trunk.sh)
        - Linux x86-64 with AVX2 + FMA
    """

    DEFAULT_BINARY = "third_party/k3/bin/k3"

    def __init__(
        self,
        binary_path: Optional[str] = None,
        model_dir: Optional[str] = None,
        trunk_dir: Optional[str] = None,
        tok_dir: Optional[str] = None,
        preset: str = "laptop",
    ) -> None:
        self.binary = self._find_binary(binary_path)
        self.model_dir = model_dir
        self.trunk_dir = trunk_dir
        self.tok_dir = tok_dir or model_dir
        self.preset = preset
        self._last_peak_rss = None
        self._last_tokens_per_sec = None

    def _find_binary(self, path: Optional[str]) -> str:
        """Locate the k3 binary."""
        if path and os.path.isfile(path):
            return path
        # Check project-local location
        local = Path(__file__).resolve().parent.parent / self.DEFAULT_BINARY
        if local.is_file():
            return str(local)
        # Check PATH
        which = shutil.which("k3")
        if which:
            return which
        return str(local)  # Return expected path for error messages

    def check_health(self) -> Dict[str, Any]:
        """
        Verify the backend is ready to run.
        Returns dict with 'ready' bool and 'issues' list.
        """
        issues = []
        
        if not os.path.isfile(self.binary):
            issues.append(f"k3 binary not found at: {self.binary}")
            issues.append("Build with: git clone https://github.com/FareedKhan-dev/kimi-k3-in-c && cd kimi-k3-in-c && make -j")
        
        if self.model_dir and not os.path.isdir(self.model_dir):
            issues.append(f"Model directory not found: {self.model_dir}")
        elif self.model_dir:
            config = Path(self.model_dir) / "config.json"
            if not config.is_file():
                issues.append(f"No config.json in model directory: {self.model_dir}")
        
        if self.trunk_dir and not os.path.isdir(self.trunk_dir):
            issues.append(f"Trunk directory not found: {self.trunk_dir}")
        
        if sys.platform != "linux":
            issues.append(f"kimi-k3-in-c requires Linux (current: {sys.platform})")

        return {
            "ready": len(issues) == 0,
            "issues": issues,
            "binary": self.binary,
            "preset": self.preset,
        }

    def generate(
        self,
        prompt: str,
        gen_tokens: int = 64,
        incremental: bool = True,
        timeout: int = 600,
    ) -> str:
        """
        Run k3 binary with the given prompt and return generated text.
        
        Args:
            prompt: Text prompt for generation.
            gen_tokens: Number of tokens to generate.
            incremental: Use stateful incremental decoding.
            timeout: Subprocess timeout in seconds (default 10 min).
            
        Returns:
            Generated text string.
            
        Raises:
            RuntimeError: If the binary fails or is not available.
        """
        if not os.path.isfile(self.binary):
            raise RuntimeError(
                f"k3 binary not found at: {self.binary}\n"
                f"Build it: git clone https://github.com/FareedKhan-dev/kimi-k3-in-c.git\n"
                f"          cd kimi-k3-in-c && make -j && make test"
            )

        cmd = [self.binary, self.model_dir]
        
        if self.trunk_dir:
            cmd.extend(["--trunk", self.trunk_dir])
        
        if self.tok_dir:
            cmd.extend(["--tok", self.tok_dir])
        
        cmd.extend([
            "--preset", self.preset,
            "--prompt", prompt,
            "--gen", str(gen_tokens),
        ])
        
        if incremental:
            cmd.append("--incremental")

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"k3 binary timed out after {timeout}s")
        except FileNotFoundError:
            raise RuntimeError(f"k3 binary not executable: {self.binary}")

        if res.returncode not in (0, 4):  # 4 = expert load failure (partial)
            raise RuntimeError(
                f"k3 binary exited with code {res.returncode}:\n{res.stderr}"
            )

        # Parse generated text from output
        generated = self._parse_output(res.stdout)
        
        # Parse memory report
        self._parse_memory_report(res.stdout)
        
        if res.returncode == 4:
            # Expert load failures — output is corrupt but we report it
            raise RuntimeError(
                f"k3 run had expert load failures (exit 4). Output is CORRUPT.\n"
                f"{res.stderr}"
            )

        return generated

    def _parse_output(self, stdout: str) -> str:
        """Extract generated text from k3 binary output."""
        # Look for "--- generated text ---" ... "----------------------"
        match = re.search(
            r"--- generated text ---\s*\n(.*?)\n----------------------",
            stdout,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        
        # Fallback: return last non-empty line before the stats
        lines = stdout.strip().splitlines()
        for i, line in enumerate(lines):
            if "tokens in" in line and "s/token" in line:
                # Return everything between "---" markers or before this line
                text_lines = []
                for j in range(i - 1, -1, -1):
                    if lines[j].strip().startswith("---"):
                        break
                    text_lines.insert(0, lines[j])
                return "\n".join(text_lines).strip()
        
        return stdout.strip()

    def _parse_memory_report(self, stdout: str) -> None:
        """Parse PEAK RSS and tokens/sec from k3 output."""
        # Parse peak RSS
        rss_match = re.search(r"PEAK RSS.*?:\s*([\d.]+)\s*GB", stdout)
        if rss_match:
            self._last_peak_rss = float(rss_match.group(1))
        
        # Parse tokens/sec
        tok_match = re.search(r"(\d+)\s*tokens?\s+in\s+([\d.]+)\s*s,\s*([\d.]+)\s*s/token", stdout)
        if tok_match:
            n_tokens = int(tok_match.group(1))
            total_sec = float(tok_match.group(2))
            if total_sec > 0:
                self._last_tokens_per_sec = n_tokens / total_sec

    def get_memory_report(self) -> Dict[str, Any]:
        """Return memory and performance stats from the last run."""
        return {
            "peak_rss_gb": self._last_peak_rss,
            "tokens_per_sec": self._last_tokens_per_sec,
        }


def find_k3_binary() -> Optional[str]:
    """Search for the k3 binary in common locations."""
    candidates = [
        Path("third_party/k3/bin/k3"),
        Path("bin/k3"),
        Path.home() / "kimi-k3-in-c" / "bin" / "k3",
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    which = shutil.which("k3")
    return which
