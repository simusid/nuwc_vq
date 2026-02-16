#!/usr/bin/env python3
"""
Train Product-Quantized VQ-VAE (Option B) on hydrophone audio.

Minimal dependencies: torch. Optional: torchaudio or soundfile for audio IO.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuwc_vq.audio import find_audio_files, load_audio, resample_if_needed  # noqa: E402
from nuwc_vq.models import PQVQVAE, PQVQVAEConfig  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class AudioSegmentDataset(Dataset):
    def __init__(
        self,
        root: Path,
        sample_rate: int,
        segment_seconds: float,
        exts: Tuple[str, ...] = (".wav", ".mp3", ".flac"),
        normalize: bool = True,
        skip_mismatch: bool = False,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.normalize = normalize
        self.skip_mismatch = skip_mismatch
        self.files = find_audio_files(root, exts)
        if not self.files:
            raise FileNotFoundError(f"No audio files found in {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        path = self.files[idx]
        wav, sr = load_audio(path)

        if sr != self.sample_rate:
            if self.skip_mismatch:
                # Return silence if skipping mismatched files
                return torch.zeros(1, self.segment_samples)
            wav = resample_if_needed(wav, sr, self.sample_rate)

        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if self.normalize:
            max_val = wav.abs().max().clamp(min=1e-8)
            wav = wav / max_val

        # Random crop / pad
        total = wav.size(1)
        if total >= self.segment_samples:
            start = random.randint(0, total - self.segment_samples)
            segment = wav[:, start : start + self.segment_samples]
        else:
            pad = self.segment_samples - total
            segment = torch.nn.functional.pad(wav, (0, pad))

        return segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PQ-VQ-VAE (Option B)")
    parser.add_argument("--data-dir", type=str, default="~/data/wavs")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-seconds", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/pq_vqvae")
    parser.add_argument("--log-dir", type=str, default="./runs/pq_vqvae")
    parser.add_argument("--skip-mismatch", action="store_true")

    # Model config overrides
    parser.add_argument("--num-embeddings-1", type=int, default=4096)
    parser.add_argument("--num-embeddings-2", type=int, default=4096)
    parser.add_argument("--embedding-dim-1", type=int, default=64)
    parser.add_argument("--embedding-dim-2", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(os.path.expanduser(args.data_dir))
    dataset = AudioSegmentDataset(
        root=data_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        normalize=True,
        skip_mismatch=args.skip_mismatch,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    config = PQVQVAEConfig(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        num_embeddings_1=args.num_embeddings_1,
        num_embeddings_2=args.num_embeddings_2,
        embedding_dim_1=args.embedding_dim_1,
        embedding_dim_2=args.embedding_dim_2,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
    )
    model = PQVQVAE(config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=args.log_dir)

    print("Config:")
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    print(f"Dataset: {len(dataset)} files, segment={args.segment_seconds}s @ {args.sample_rate}Hz")

    global_step = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        for batch_idx, x in enumerate(loader, start=1):
            x = x.to(args.device)

            x_hat, stats = model(x)
            loss = model.loss(x, x_hat, stats["commitment_loss"])

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % args.log_interval == 0:
                with torch.no_grad():
                    # Unique code usage per batch
                    idx1 = stats["encoding_idx_1"].reshape(-1)
                    idx2 = stats["encoding_idx_2"].reshape(-1)
                    usage_1 = idx1.unique().numel() / args.num_embeddings_1
                    usage_2 = idx2.unique().numel() / args.num_embeddings_2

                    writer.add_scalar("train/loss", loss.item(), global_step)
                    writer.add_scalar(
                        "train/commitment_loss",
                        stats["commitment_loss"].item(),
                        global_step,
                    )
                    writer.add_scalar(
                        "train/perplexity_1", stats["perplexity_1"].item(), global_step
                    )
                    writer.add_scalar(
                        "train/perplexity_2", stats["perplexity_2"].item(), global_step
                    )
                    writer.add_scalar("train/usage_1", usage_1, global_step)
                    writer.add_scalar("train/usage_2", usage_2, global_step)

                    print(
                        " | ".join(
                            [
                                f"epoch={epoch}",
                                f"step={global_step}",
                                f"loss={loss.item():.4f}",
                                f"ppl1={stats['perplexity_1'].item():.2f}",
                                f"ppl2={stats['perplexity_2'].item():.2f}",
                                f"usage1={usage_1:.3f}",
                                f"usage2={usage_2:.3f}",
                            ]
                        )
                    )

            global_step += 1

        ckpt_path = Path(args.checkpoint_dir) / f"pq_vqvae_epoch_{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": asdict(config),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint: {ckpt_path}")

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
