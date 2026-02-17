"""
Product-quantized VQ-VAE for 1D acoustic waveforms.
Option B: two parallel EMA codebooks at the same timestep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int | None = None):
        super().__init__()
        hidden = hidden_channels or channels
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 128,
        latent_channels: int = 128,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 3

        self.conv1 = nn.Conv1d(in_channels, c1, kernel_size=7, stride=2, padding=3)
        self.res1 = nn.Sequential(*[ResBlock(c1) for _ in range(num_res_blocks)])

        self.conv2 = nn.Conv1d(c1, c2, kernel_size=5, stride=2, padding=2)
        self.res2 = nn.Sequential(*[ResBlock(c2) for _ in range(num_res_blocks)])

        self.conv3 = nn.Conv1d(c2, c3, kernel_size=3, stride=2, padding=1)
        self.res3 = nn.Sequential(*[ResBlock(c3) for _ in range(num_res_blocks)])

        self.proj = nn.Conv1d(c3, latent_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.res1(x)
        x = self.conv2(x)
        x = self.res2(x)
        x = self.conv3(x)
        x = self.res3(x)
        x = self.proj(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 1,
        base_channels: int = 128,
        latent_channels: int = 128,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        c1 = base_channels * 3
        c2 = base_channels * 2
        c3 = base_channels

        self.proj = nn.Conv1d(latent_channels, c1, kernel_size=1)
        self.res1 = nn.Sequential(*[ResBlock(c1) for _ in range(num_res_blocks)])
        self.up1 = nn.ConvTranspose1d(c1, c2, kernel_size=4, stride=2, padding=1)

        self.res2 = nn.Sequential(*[ResBlock(c2) for _ in range(num_res_blocks)])
        self.up2 = nn.ConvTranspose1d(c2, c3, kernel_size=4, stride=2, padding=1)

        self.res3 = nn.Sequential(*[ResBlock(c3) for _ in range(num_res_blocks)])
        self.up3 = nn.ConvTranspose1d(c3, c3, kernel_size=4, stride=2, padding=1)

        self.out = nn.Conv1d(c3, out_channels, kernel_size=7, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.res1(x)
        x = self.up1(x)
        x = self.res2(x)
        x = self.up2(x)
        x = self.res3(x)
        x = self.up3(x)
        x = self.out(x)
        return x


class EMAQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        dead_code_threshold: float = 0.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps
        self.commitment_cost = commitment_cost
        self.dead_code_threshold = dead_code_threshold

        embed = torch.randn(num_embeddings, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, D, T)
        b, d, t = x.shape
        x_flat = x.permute(0, 2, 1).contiguous().view(-1, d)

        # distances to codebook
        embed = self.embed
        dist = (
            x_flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * x_flat @ embed.t()
            + embed.pow(2).sum(dim=1)
        )

        encoding_idx = torch.argmin(dist, dim=1)
        encodings = F.one_hot(encoding_idx, self.num_embeddings).type(x_flat.dtype)

        quantized = encodings @ embed
        quantized = quantized.view(b, t, d).permute(0, 2, 1).contiguous()

        if self.training:
            with torch.no_grad():
                cluster_size = encodings.sum(dim=0)
                embed_sum = encodings.t() @ x_flat

                self.cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                n = self.cluster_size.sum()
                cluster_size = (
                    (self.cluster_size + self.eps)
                    / (n + self.num_embeddings * self.eps)
                    * n
                )
                embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
                self.embed.copy_(embed_normalized)

                if self.dead_code_threshold > 0.0:
                    dead = self.cluster_size < self.dead_code_threshold
                    if dead.any():
                        rand_idx = torch.randint(0, x_flat.size(0), (dead.sum(),))
                        self.embed[dead] = x_flat[rand_idx]
                        self.embed_avg[dead] = x_flat[rand_idx]
                        self.cluster_size[dead] = self.dead_code_threshold

        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), x)
        quantized = x + (quantized - x).detach()

        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        usage_loss = torch.sum(
            avg_probs * torch.log(avg_probs * self.num_embeddings + 1e-10)
        )

        stats = {
            "commitment_loss": commitment_loss,
            "perplexity": perplexity,
            "avg_probs": avg_probs,
            "encoding_idx": encoding_idx.view(b, t),
            "usage_loss": usage_loss,
        }
        return quantized, stats


class ProductEMAQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings_1: int = 4096,
        num_embeddings_2: int = 4096,
        embedding_dim_1: int = 64,
        embedding_dim_2: int = 64,
        decay: float = 0.99,
        eps: float = 1e-5,
        commitment_cost: float = 0.25,
        dead_code_threshold: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim_1 = embedding_dim_1
        self.embedding_dim_2 = embedding_dim_2

        self.q1 = EMAQuantizer(
            num_embeddings=num_embeddings_1,
            embedding_dim=embedding_dim_1,
            decay=decay,
            eps=eps,
            commitment_cost=commitment_cost,
            dead_code_threshold=dead_code_threshold,
        )
        self.q2 = EMAQuantizer(
            num_embeddings=num_embeddings_2,
            embedding_dim=embedding_dim_2,
            decay=decay,
            eps=eps,
            commitment_cost=commitment_cost,
            dead_code_threshold=dead_code_threshold,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x: (B, D, T)
        d1 = self.embedding_dim_1
        d2 = self.embedding_dim_2
        if x.size(1) != d1 + d2:
            raise ValueError(
                f"Expected {d1 + d2} channels, got {x.size(1)}. "
                "Ensure latent_channels = embedding_dim_1 + embedding_dim_2."
            )

        x1, x2 = torch.split(x, [d1, d2], dim=1)
        q1, s1 = self.q1(x1)
        q2, s2 = self.q2(x2)

        quantized = torch.cat([q1, q2], dim=1)
        stats = {
            "commitment_loss": s1["commitment_loss"] + s2["commitment_loss"],
            "perplexity_1": s1["perplexity"],
            "perplexity_2": s2["perplexity"],
            "avg_probs_1": s1["avg_probs"],
            "avg_probs_2": s2["avg_probs"],
            "encoding_idx_1": s1["encoding_idx"],
            "encoding_idx_2": s2["encoding_idx"],
            "usage_loss": s1["usage_loss"] + s2["usage_loss"],
            "usage_loss_1": s1["usage_loss"],
            "usage_loss_2": s2["usage_loss"],
        }
        return quantized, stats


@dataclass
class PQVQVAEConfig:
    in_channels: int = 1
    out_channels: int = 1
    base_channels: int = 128
    latent_channels: int = 128
    num_res_blocks: int = 2
    num_embeddings_1: int = 4096
    num_embeddings_2: int = 4096
    embedding_dim_1: int = 64
    embedding_dim_2: int = 64
    decay: float = 0.99
    commitment_cost: float = 0.25
    usage_regularizer_weight: float = 0.1
    dead_code_threshold: float = 1.0


class PQVQVAE(nn.Module):
    def __init__(self, config: PQVQVAEConfig):
        super().__init__()
        if config.embedding_dim_1 + config.embedding_dim_2 != config.latent_channels:
            raise ValueError(
                "latent_channels must equal embedding_dim_1 + embedding_dim_2"
            )

        self.encoder = Encoder(
            in_channels=config.in_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            num_res_blocks=config.num_res_blocks,
        )
        self.quantizer = ProductEMAQuantizer(
            num_embeddings_1=config.num_embeddings_1,
            num_embeddings_2=config.num_embeddings_2,
            embedding_dim_1=config.embedding_dim_1,
            embedding_dim_2=config.embedding_dim_2,
            decay=config.decay,
            commitment_cost=config.commitment_cost,
            dead_code_threshold=config.dead_code_threshold,
        )
        self.decoder = Decoder(
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            latent_channels=config.latent_channels,
            num_res_blocks=config.num_res_blocks,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        z = self.encoder(x)
        q, stats = self.quantizer(z)
        x_hat = self.decoder(q)
        return x_hat, stats

    @staticmethod
    def loss(
        x: torch.Tensor,
        x_hat: torch.Tensor,
        commitment_loss: torch.Tensor,
        usage_loss: torch.Tensor | None = None,
        usage_weight: float = 0.0,
        recon_weight: float = 1.0,
    ) -> torch.Tensor:
        recon = F.l1_loss(x_hat, x)
        total = recon_weight * recon + commitment_loss
        if usage_loss is not None and usage_weight > 0.0:
            total = total + usage_weight * usage_loss
        return total
