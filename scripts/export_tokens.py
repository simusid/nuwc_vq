#!/usr/bin/env python3
"""
Export PQ-VQ-VAE token sequences from audio files.
Saves two parallel codebook index sequences per segment.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuwc_vq.audio import find_audio_files, load_audio, resample_if_needed  # noqa: E402
from nuwc_vq.models import PQVQVAE, PQVQVAEConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PQ-VQ-VAE tokens")
    parser.add_argument("--data-dir", type=str, default="~/data/wavs")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./tokens/pq_vqvae")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--exts", type=str, default=".wav,.mp3,.flac")
    return parser.parse_args()


def chunk_audio(wav: torch.Tensor, segment_samples: int) -> torch.Tensor:
    # wav: (1, T)
    total = wav.size(1)
    if total <= segment_samples:
        pad = segment_samples - total
        return torch.nn.functional.pad(wav, (0, pad)).unsqueeze(0)

    num_segments = (total + segment_samples - 1) // segment_samples
    padded = torch.nn.functional.pad(wav, (0, num_segments * segment_samples - total))
    segments = padded.view(1, num_segments, segment_samples).transpose(0, 1)
    return segments


def load_model(ckpt_path: Path, device: str) -> PQVQVAE:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    config_dict = ckpt.get("config", {})
    config = PQVQVAEConfig(**config_dict)
    model = PQVQVAE(config)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()

    data_dir = Path(os.path.expanduser(args.data_dir))
    out_dir = Path(os.path.expanduser(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = tuple([e.strip() for e in args.exts.split(",") if e.strip()])
    files = find_audio_files(data_dir, exts)
    if not files:
        raise FileNotFoundError(f"No audio files found in {data_dir}")

    model = load_model(Path(args.checkpoint), args.device)

    segment_samples = int(args.sample_rate * args.segment_seconds)

    for path in files:
        wav, sr = load_audio(path)
        if sr != args.sample_rate:
            wav = resample_if_needed(wav, sr, args.sample_rate)

        # mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if args.normalize:
            max_val = wav.abs().max().clamp(min=1e-8)
            wav = wav / max_val

        segments = chunk_audio(wav, segment_samples)
        # segments: (N, 1, segment_samples)

        tokens_1 = []
        tokens_2 = []
        with torch.no_grad():
            for seg in segments:
                seg = seg.unsqueeze(0).to(args.device)
                _, stats = model(seg)
                tokens_1.append(stats["encoding_idx_1"].cpu())
                tokens_2.append(stats["encoding_idx_2"].cpu())

        tokens_1 = torch.cat(tokens_1, dim=0)
        tokens_2 = torch.cat(tokens_2, dim=0)

        rel = path.relative_to(data_dir)
        out_path = out_dir / rel
        out_path = out_path.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "path": str(path),
                "sample_rate": args.sample_rate,
                "segment_seconds": args.segment_seconds,
                "tokens_1": tokens_1,
                "tokens_2": tokens_2,
            },
            out_path,
        )
        print(f"Saved tokens: {out_path}")


if __name__ == "__main__":
    main()
