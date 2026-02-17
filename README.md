# NUWC VQ

Tokenizer-first pipeline for undersea hydrophone acoustics. This project trains a vector-quantized variational autoencoder (VQ-VAE) to build a discrete codebook for diverse acoustic signals. The learned tokens are then used to train a large-scale acoustic model (LAM) with LLM-like structure and scale. The first, critical component is the companion tokenizer.

## Goals
- Build a robust VQ-VAE tokenizer for undersea acoustic signals across seasons, locations, sampling rates, and signal types.
- Emphasize man-made signal detection while retaining coverage of natural and environmental acoustics for generalization.
- Produce high-quality, stable token sequences suitable for training a large acoustic model.

## Signal Domains
The dataset is heterogeneous and may include:
- Sea-state wind and wave noise.
- Marine biologics (e.g., whales, whale calls).
- Geologic low-frequency rumbles.
- Broadband and narrowband emissions.
- Impulsive events (e.g., snapping shrimp).
- Complex structured signals (e.g., whale calls).
- Marine traffic and shipping noise.
- Metallic transients and other man-made signatures.

The primary interest is in detection of man-made signals; natural signals provide context and robustness.

## Tokenizer Strategy
The VQ-VAE is trained to discretize acoustic signals into codebook indices:
- One or more codebooks (typically two).
- Continuous audio is mapped to discrete tokens.
- Tokens are used as the vocabulary for downstream LAM training.

## Data Assumptions
- Inputs are hydrophone recordings (wav or mp3).
- Default local data path: `~/data/wavs`.
- Typical sampling rate: `16 kHz` (but multiple rates are expected).
- Signals vary across time of year, geography, and operational systems.

## Repository Usage (High-Level)
This README is meant to be comprehensive and will evolve with the codebase. As code is added, update these sections with concrete commands, configs, and expected outputs.

Planned workflows:
- Ingest and validate diverse audio sources.
- Normalize sampling rates and segment audio.
- Train VQ-VAE to learn codebooks.
- Quantize audio to tokens.
- Train a large acoustic model using token sequences.
- Evaluate detection performance on man-made signals.

## Evaluation Focus
Success is measured by:
- Token stability across variable conditions.
- Codebook coverage across signal types.
- Detection performance on man-made events and signatures.
- Generalization across time, location, and sensor systems.

## Roadmap
- Define data schema and metadata standards (time, location, sensor, sample rate, labeling).
- Implement preprocessing and normalization pipelines.
- Add training and evaluation scripts for VQ-VAE.
- Add tokenization pipeline and storage format.
- Integrate downstream LAM training pipeline.
- Establish detection benchmarks and metrics.

## Notes
This project is intentionally oriented toward detection and tokenization quality. The tokenizer is treated as a foundational artifact for all downstream acoustic modeling.

## Implemented Model: Option B (Product-Quantized VQ-VAE)
The project now includes a Product-Quantized VQ-VAE with two parallel EMA codebooks at each timestep.

Location:
- `/Users/gary/Desktop/nuwc_vq/nuwc_vq/models/pq_vqvae.py`

Key defaults:
- Two codebooks, each `4096` entries
- Embedding dims `64 + 64` (total latent channels `128`)
- EMA decay `0.99`, commitment loss `0.25`
- Encoder/decoder are 1D conv stacks (3x stride-2 downsamples)

Minimal usage:
```python
import torch
from nuwc_vq.models import PQVQVAE, PQVQVAEConfig

config = PQVQVAEConfig()
model = PQVQVAE(config)

x = torch.randn(2, 1, 16000)  # 1 second @ 16 kHz
x_hat, stats = model(x)
loss = model.loss(x, x_hat, stats["commitment_loss"])
```

## Training Script
A minimal training script is included for the Product-Quantized VQ-VAE.

Location:
- `/Users/gary/Desktop/nuwc_vq/scripts/train_pq_vqvae.py`

Example:
```bash
python /Users/gary/Desktop/nuwc_vq/scripts/train_pq_vqvae.py \
  --data-dir ~/data/wavs \
  --sample-rate 16000 \
  --segment-seconds 1.0 \
  --batch-size 16 \
  --epochs 10
```

Notes:
- Uses `torchaudio` if available for loading and resampling; falls back to `soundfile` for wav/flac.
- MP3 support requires `torchaudio`.
- Logs per-batch codebook usage and perplexity to validate token utilization.
- Checkpoints are saved to `./checkpoints/pq_vqvae` by default.
- Shows a progress bar while indexing files (disable with `--no-index-progress`).
- If you see DataLoader stalls with `num_workers > 0`, use `--mp-context spawn` (default) or set `--num-workers 0`.

TensorBoard:
```bash
tensorboard --logdir ./runs/pq_vqvae
```

## Token Export Script
Export token sequences (two parallel codebooks) for downstream LAM training.

Location:
- `/Users/gary/Desktop/nuwc_vq/scripts/export_tokens.py`

Example:
```bash
python /Users/gary/Desktop/nuwc_vq/scripts/export_tokens.py \
  --data-dir ~/data/wavs \
  --checkpoint ./checkpoints/pq_vqvae/pq_vqvae_epoch_10.pt \
  --output-dir ./tokens/pq_vqvae \
  --sample-rate 16000 \
  --segment-seconds 1.0
```

Output:
- Mirrors input folder structure under `./tokens/pq_vqvae`
- Each file saves `.pt` with `tokens_1`, `tokens_2`, and metadata
- Shows a progress bar while indexing files (disable with `--no-index-progress`).

## Requirements
```bash
pip install -r /Users/gary/Desktop/nuwc_vq/requirements.txt
```
