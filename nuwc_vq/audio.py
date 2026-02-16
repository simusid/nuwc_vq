"""
Audio IO utilities for NUWC VQ.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch


def find_audio_files(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def load_audio(path: Path) -> Tuple[torch.Tensor, int]:
    ext = path.suffix.lower()

    try:
        import torchaudio  # type: ignore

        wav, sr = torchaudio.load(str(path))
        return wav, sr
    except Exception:
        pass

    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), always_2d=True)
        wav = torch.from_numpy(data.T).float()
        return wav, sr
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {path} (ext={ext}). "
            "Install torchaudio for mp3 support or soundfile for wav/flac."
        ) from exc


def resample_if_needed(wav: torch.Tensor, sr: int, target_sr: int) -> torch.Tensor:
    if sr == target_sr:
        return wav
    try:
        import torchaudio  # type: ignore

        return torchaudio.functional.resample(wav, sr, target_sr)
    except Exception as exc:
        raise RuntimeError(
            f"Resample needed ({sr} -> {target_sr}) but torchaudio not available."
        ) from exc
