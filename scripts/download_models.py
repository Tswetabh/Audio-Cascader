"""Download script for offline STT and TTS models."""

from __future__ import annotations

import os
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def download_file(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url} -> {dest}...")
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, str(dest))
    print(f"Downloaded: {dest} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")

def download_piper():
    print("\n--- 1. Piper Voice (en_US-lessac-medium) ---")
    dest_dir = ROOT / "tts" / "piper" / "models"
    user_dir = Path.home() / ".local" / "share" / "piper"
    dest_dir.mkdir(parents=True, exist_ok=True)
    user_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium"
    model_name = "en_US-lessac-medium.onnx"
    json_name = "en_US-lessac-medium.onnx.json"

    for fname in [model_name, json_name]:
        url = f"{base_url}/{fname}"
        target = dest_dir / fname
        download_file(url, target)
        # Also copy / symlink to user_dir
        user_target = user_dir / fname
        if not user_target.exists():
            import shutil
            shutil.copyfile(target, user_target)

def download_zipformer():
    print("\n--- 2. Sherpa-ONNX Zipformer (20M streaming) ---")
    dest_dir = ROOT / "stt" / "zipformer"
    dest_dir.mkdir(parents=True, exist_ok=True)
    model_folder = dest_dir / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
    if model_folder.exists() and (model_folder / "tokens.txt").exists():
        print(f"Zipformer already present: {model_folder}")
        return

    tar_url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
    tar_path = dest_dir / "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2"
    download_file(tar_url, tar_path)

    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:bz2") as tar:
        tar.extractall(path=dest_dir)
    print("Extracted Zipformer successfully!")

def download_kokoro_onnx():
    print("\n--- 3. Kokoro ONNX Models ---")
    dest_dir = ROOT / "tts" / "kokoro" / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Download kokoro-v1.0.onnx and voices-v1.0.bin
    files = {
        "kokoro-v1.0.onnx": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
        "voices-v1.0.bin": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    }
    for fname, url in files.items():
        target = dest_dir / fname
        try:
            download_file(url, target)
        except Exception as e:
            print(f"Warning: could not download {fname}: {e}")

if __name__ == "__main__":
    download_piper()
    download_zipformer()
    download_kokoro_onnx()
    print("\nDone downloading models!")
