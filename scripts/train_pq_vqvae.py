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
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nuwc_vq.audio import find_audio_files, load_audio, resample_if_needed  # noqa: E402
from nuwc_vq.models import PQVQVAE, PQVQVAEConfig  # noqa: E402


DEFAULT_EXTS: Tuple[str, ...] = (".wav", ".mp3", ".flac")


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
        exts: Tuple[str, ...] = DEFAULT_EXTS,
        normalize: bool = True,
        skip_mismatch: bool = False,
        index_progress: bool = True,
        files: List[Path] | None = None,
    ):
        self.root = root
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_seconds)
        self.normalize = normalize
        self.skip_mismatch = skip_mismatch
        self.files = files or find_audio_files(root, exts, show_progress=index_progress)
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
    parser.add_argument("--no-index-progress", action="store_true")
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--val-dir", type=str, default="")
    parser.add_argument("--val-split", type=float, default=0.0)
    parser.add_argument("--val-max-batches", type=int, default=0)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument(
        "--mp-context",
        type=str,
        default="spawn",
        choices=["spawn", "fork", "forkserver", "none"],
        help="Multiprocessing context for DataLoader when num_workers > 0.",
    )
    parser.add_argument(
        "--loader-timeout",
        type=int,
        default=0,
        help="Seconds to wait for a batch before timing out (0 = no timeout).",
    )

    # Model config overrides
    parser.add_argument("--num-embeddings-1", type=int, default=4096)
    parser.add_argument("--num-embeddings-2", type=int, default=4096)
    parser.add_argument("--embedding-dim-1", type=int, default=64)
    parser.add_argument("--embedding-dim-2", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--commitment-cost", type=float, default=0.25)
    parser.add_argument("--decay", type=float, default=0.99)
    parser.add_argument("--usage-regularizer-weight", type=float, default=0.1)
    parser.add_argument("--dead-code-threshold", type=float, default=1.0)

    return parser.parse_args()


def split_train_val(
    files: List[Path], val_split: float, seed: int
) -> Tuple[List[Path], List[Path]]:
    if val_split <= 0:
        return files, []
    rng = random.Random(seed)
    files = list(files)
    rng.shuffle(files)
    n_val = max(1, int(len(files) * val_split))
    return files[n_val:], files[:n_val]


def evaluate(
    model: PQVQVAE,
    loader: DataLoader,
    device: str,
    usage_weight: float,
    max_batches: int = 0,
) -> dict:
    model.eval()
    totals = {
        "loss": 0.0,
        "recon": 0.0,
        "commit": 0.0,
        "usage": 0.0,
        "ppl1": 0.0,
        "ppl2": 0.0,
        "count": 0,
    }
    with torch.no_grad():
        for i, x in enumerate(loader):
            if max_batches and i >= max_batches:
                break
            x = x.to(device)
            x_hat, stats = model(x)
            recon = F.l1_loss(x_hat, x)
            commit = stats["commitment_loss"]
            usage = stats.get("usage_loss", torch.tensor(0.0, device=x.device))
            total = recon + commit + usage_weight * usage

            totals["loss"] += total.item()
            totals["recon"] += recon.item()
            totals["commit"] += commit.item()
            totals["usage"] += usage.item()
            totals["ppl1"] += stats["perplexity_1"].item()
            totals["ppl2"] += stats["perplexity_2"].item()
            totals["count"] += 1

    model.train()
    if totals["count"] == 0:
        return {}
    return {k: v / totals["count"] for k, v in totals.items() if k != "count"}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(os.path.expanduser(args.data_dir))
    all_files = find_audio_files(
        data_dir, DEFAULT_EXTS, show_progress=not args.no_index_progress
    )
    if not all_files:
        raise FileNotFoundError(f"No audio files found in {data_dir}")

    val_files: List[Path] = []
    if args.val_dir:
        val_dir = Path(os.path.expanduser(args.val_dir))
        val_files = find_audio_files(val_dir, DEFAULT_EXTS, show_progress=False)
        val_set = {p.resolve() for p in val_files}
        train_files = [p for p in all_files if p.resolve() not in val_set]
    else:
        train_files, val_files = split_train_val(
            all_files, args.val_split, args.seed
        )

    dataset = AudioSegmentDataset(
        root=data_dir,
        sample_rate=args.sample_rate,
        segment_seconds=args.segment_seconds,
        normalize=True,
        skip_mismatch=args.skip_mismatch,
        index_progress=False,
        files=train_files,
    )
    mp_context = None
    if args.num_workers > 0 and args.mp_context != "none":
        try:
            mp.set_start_method(args.mp_context, force=True)
        except RuntimeError:
            pass
        mp_context = mp.get_context(args.mp_context)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "drop_last": True,
        "pin_memory": not args.no_pin_memory,
        "multiprocessing_context": mp_context,
        "timeout": args.loader_timeout,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(**loader_kwargs)

    val_loader = None
    if val_files:
        val_dataset = AudioSegmentDataset(
            root=data_dir,
            sample_rate=args.sample_rate,
            segment_seconds=args.segment_seconds,
            normalize=True,
            skip_mismatch=args.skip_mismatch,
            index_progress=False,
            files=val_files,
        )
        val_loader_kwargs = {
            "dataset": val_dataset,
            "batch_size": args.batch_size,
            "shuffle": False,
            "num_workers": args.num_workers,
            "drop_last": False,
            "pin_memory": not args.no_pin_memory,
            "multiprocessing_context": mp_context,
            "timeout": args.loader_timeout,
        }
        if args.num_workers > 0:
            val_loader_kwargs["prefetch_factor"] = args.prefetch_factor
        val_loader = DataLoader(**val_loader_kwargs)

    config = PQVQVAEConfig(
        base_channels=args.base_channels,
        latent_channels=args.latent_channels,
        num_embeddings_1=args.num_embeddings_1,
        num_embeddings_2=args.num_embeddings_2,
        embedding_dim_1=args.embedding_dim_1,
        embedding_dim_2=args.embedding_dim_2,
        commitment_cost=args.commitment_cost,
        decay=args.decay,
        usage_regularizer_weight=args.usage_regularizer_weight,
        dead_code_threshold=args.dead_code_threshold,
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
    print(
        f"Dataset: {len(dataset)} train files, {len(val_files)} val files, "
        f"segment={args.segment_seconds}s @ {args.sample_rate}Hz"
    )

    global_step = 0
    model.train()
    best_metric = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        for batch_idx, x in enumerate(loader, start=1):
            x = x.to(args.device)

            x_hat, stats = model(x)
            loss = model.loss(
                x,
                x_hat,
                stats["commitment_loss"],
                usage_loss=stats.get("usage_loss"),
                usage_weight=config.usage_regularizer_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if global_step % args.log_interval == 0:
                with torch.no_grad():
                    # Unique code usage per batch
                    idx1 = stats["encoding_idx_1"].reshape(-1)
                    idx2 = stats["encoding_idx_2"].reshape(-1)
                    used_1 = idx1.unique().numel()
                    used_2 = idx2.unique().numel()
                    usage_1 = used_1 / args.num_embeddings_1
                    usage_2 = used_2 / args.num_embeddings_2

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
                    if "usage_loss_1" in stats:
                        writer.add_scalar(
                            "train/usage_loss_1",
                            stats["usage_loss_1"].item(),
                            global_step,
                        )
                    if "usage_loss_2" in stats:
                        writer.add_scalar(
                            "train/usage_loss_2",
                            stats["usage_loss_2"].item(),
                            global_step,
                        )

                    print(
                        " | ".join(
                            [
                                f"epoch={epoch}",
                                f"step={global_step}",
                                f"loss={loss.item():.4f}",
                                f"ppl1={stats['perplexity_1'].item():.2f}",
                                f"ppl2={stats['perplexity_2'].item():.2f}",
                                f"usage1={usage_1:.5f}({used_1})",
                                f"usage2={usage_2:.5f}({used_2})",
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

        if val_loader is not None:
            val_stats = evaluate(
                model,
                val_loader,
                args.device,
                config.usage_regularizer_weight,
                max_batches=args.val_max_batches,
            )
            if val_stats:
                writer.add_scalar("val/loss", val_stats["loss"], epoch)
                writer.add_scalar("val/recon", val_stats["recon"], epoch)
                writer.add_scalar("val/commitment", val_stats["commit"], epoch)
                writer.add_scalar("val/usage_loss", val_stats["usage"], epoch)
                writer.add_scalar("val/perplexity_1", val_stats["ppl1"], epoch)
                writer.add_scalar("val/perplexity_2", val_stats["ppl2"], epoch)

                print(
                    " | ".join(
                        [
                            f"val_epoch={epoch}",
                            f"val_loss={val_stats['loss']:.4f}",
                            f"val_recon={val_stats['recon']:.4f}",
                            f"val_commit={val_stats['commit']:.4f}",
                            f"val_usage={val_stats['usage']:.4f}",
                            f"val_ppl1={val_stats['ppl1']:.2f}",
                            f"val_ppl2={val_stats['ppl2']:.2f}",
                        ]
                    )
                )

                if val_stats["loss"] < best_metric - args.min_delta:
                    best_metric = val_stats["loss"]
                    epochs_no_improve = 0
                    best_path = Path(args.checkpoint_dir) / "pq_vqvae_best.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": asdict(config),
                            "val": val_stats,
                        },
                        best_path,
                    )
                    print(f"Saved best checkpoint: {best_path}")
                else:
                    epochs_no_improve += 1

                if args.early_stop and epochs_no_improve >= args.patience:
                    print(
                        f"Early stopping triggered at epoch {epoch} "
                        f"(no improvement in {args.patience} epochs)."
                    )
                    break

        if args.early_stop and val_loader is not None and epochs_no_improve >= args.patience:
            break

    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
