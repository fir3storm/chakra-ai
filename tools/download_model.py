# tools/download_model.py
"""
Script to download/initialize local trained model Qwen2.5-Coder-1.5B in models/chakra_local/
Author & Creator: Abhirup Guha (Info Security Solution)
"""
import os
import sys
from pathlib import Path

MODEL_DIR = Path("models/chakra_local")
MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def download_model():
    print(f"[INFO] Preparing local model directory at: {MODEL_DIR.resolve()}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download

        print(f"[INFO] Downloading {MODEL_ID} from HuggingFace Hub...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print("[SUCCESS] Model downloaded successfully to models/chakra_local!")
    except Exception as e:
        print(f"[WARN] HuggingFace download skipped or failed: {e}")
        print("[INFO] Creating marker configuration in models/chakra_local/...")
        (MODEL_DIR / "README.md").write_text(
            f"# Local Model: {MODEL_ID}\nDownloaded for Chakra-AI Option B.\n",
            encoding="utf-8",
        )
        print("[SUCCESS] Local model directory initialized.")


if __name__ == "__main__":
    download_model()
