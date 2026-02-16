"""
Audio IO utilities for NUWC VQ.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Tuple

import torch


def _iter_audio_paths(root: Path, exts: Tuple[str, ...]) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                yield Path(dirpath) / name


def find_audio_files(
    root: Path, exts: Tuple[str, ...], show_progress: bool = False
) -> List[Path]:
    if show_progress:
        try:
            from tqdm import tqdm  # type: ignore

            files: List[Path] = []
            with tqdm(
                total=0,
                unit="files",
                desc="Indexing audio files",
                leave=True,
            ) as pbar:
                for path in _iter_audio_paths(root, exts):
                    files.append(path)
                    pbar.update(1)
            return sorted(files)
        except Exception:
            pass

    files = [p for p in _iter_audio_paths(root, exts)]
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
