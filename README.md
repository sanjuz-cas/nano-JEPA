# Nano JEPA

A minimal, high-performance PyTorch implementation of **JEPA** (Joint-Embedding Predictive Architecture) for self-supervised visual representation learning.

Trained and evaluated on the [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells) dataset with support for any `ImageFolder`-compatible dataset.

---

## Architecture

Nano JEPA learns visual representations by predicting masked patch embeddings in latent space — without pixel-level reconstruction.

```text
                    ┌─────────────────────┐
                    │   Input Image (x)   │
                    └────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌────────────────┐           ┌────────────────┐
     │  Patch Embed   │           │ Target Patch   │
     │  + Pos Embed   │           │ Embed + Pos    │
     └───────┬────────┘           └───────┬────────┘
             │                            │
        Random Mask                  No masking
        (75% masked)                     │
             │                            ▼
             ▼                   ┌────────────────┐
     ┌────────────────┐          │ Target Encoder │ ← EMA updated
     │Context Encoder │          │  (frozen)      │    (no gradients)
     │(visible only)  │          └───────┬────────┘
     └───────┬────────┘                  │
             │                           │
             ▼                           ▼
     ┌────────────────┐          ┌──────────────┐
     │   Predictor    │ ──loss──▶│ Target feats │
     │  (full seq)    │          │ (masked pos) │
     └────────────────┘          └──────────────┘
```

**Key idea**: The context encoder sees only the visible (unmasked) patches. The predictor reconstructs the *latent representations* of masked patches, supervised by the EMA target encoder's output. This forces the model to learn semantically meaningful features without relying on pixel-level details.

---

## Model Configuration

| Component | Default | Scaled-Up |
|---|---:|---:|
| Input Resolution | 64 × 64 | 224 × 224 |
| Patch Size | 8 × 8 | 16 × 16 |
| Total Patches | 64 | 196 |
| Embedding Dimension | 384 | 768 |
| Context Encoder Layers | 6 | 12 |
| Target Encoder Layers | 6 | 12 |
| Predictor Layers | 3 | 4 |
| Attention Heads | 12 | 12 |
| Total Parameters | ~26.96M | ~200.54M |
| Trainable Parameters | ~15.39M | ~114.74M |
| Mask Ratio | 75% | 75% |
| Optimizer | AdamW | AdamW |
| Precision | FP16 mixed | FP16 mixed |

---

## Repository Structure

```text
nano-jepa/
├── config.py          # Dataclass-based configuration and CLI argument parser
├── model.py           # NanoJEPA model, Transformer blocks, EMA target, JEPA loss
├── data.py            # ImageFolder dataset, synthetic benchmarking dataset, DataLoaders
├── train.py           # Training loop, DDP, MFU logging, checkpointing, t-SNE export
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT License
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/nano-jepa.git
cd nano-jepa
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.3.0
- torchvision ≥ 0.18.0
- NVIDIA GPU with CUDA support

For t-SNE visualization (optional):

```bash
pip install scikit-learn matplotlib
```

---

## Dataset Setup

This project uses `torchvision.datasets.ImageFolder`, which expects a directory with class subfolders:

```text
dataset/
└── TRAIN/
    ├── EOSINOPHIL/
    │   ├── _0_1234.jpeg
    │   ├── _0_5678.jpeg
    │   └── ...
    ├── LYMPHOCYTE/
    ├── MONOCYTE/
    └── NEUTROPHIL/
```

**Using the Blood Cell Images dataset:**

1. Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/blood-cells)
2. Extract and point `--data-path` to the folder containing class subfolders

---

## Quick Start

### Single GPU — Default Config

```bash
python train.py \
    --data-path ./path/to/TRAIN \
    --batch-size 256 \
    --epochs 20
```

### Single GPU — Scaled-Up Config (≥16 GB VRAM)

```bash
python train.py \
    --data-path ./path/to/TRAIN \
    --img-size 224 --patch-size 16 \
    --embed-dim 768 --context-depth 12 --target-depth 12 --predictor-depth 4 \
    --heads 12 --mlp-ratio 4.0 \
    --batch-size 128 \
    --compile \
    --disable-grad-checkpointing \
    --flops-per-image 48e9
```

### Multi-GPU (DDP)

```bash
OMP_NUM_THREADS=4 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --data-path ./path/to/TRAIN \
    --batch-size 256 \
    --epochs 20
```

### Kaggle Notebook

```python
!python train.py \
    --data-path "/kaggle/input/blood-cells/dataset2-master/dataset2-master/images/TRAIN" \
    --img-size 224 --patch-size 16 \
    --embed-dim 768 --context-depth 12 --target-depth 12 --predictor-depth 4 \
    --batch-size 128 --compile --disable-grad-checkpointing \
    --flops-per-image 48e9
```

---

## Training Output

During training, the logger prints per-step metrics:

```text
================================================================================
Nano JEPA HPC Trainer
================================================================================
DDP enabled          : False
World size           : 1
Per-GPU batch size   : 128
Global batch size    : 128
FP16                 : True
torch.compile        : True
Total parameters     : 200.54M
Trainable parameters : 114.74M
================================================================================
[Epoch 000][Step 000019] loss=0.1940 lr=0.000039 mom=0.99600 img/sec=122.0 MFU=9.007%
[Epoch 000][Step 000039] loss=0.0813 lr=0.000078 mom=0.99601 img/sec=133.5 MFU=9.857%
[Epoch 001][Final]       img/sec=140.7 TFLOPS=6.754 MFU=10.391%
```

Checkpoints are saved to `./outputs/` every `--save-every` epochs:

```text
outputs/
├── nano_jepa_epoch_004.pth
├── nano_jepa_epoch_009.pth
├── nano_jepa_epoch_014.pth
└── nano_jepa_epoch_019.pth
```

---

## t-SNE Visualization

Generate a t-SNE plot of the learned representations after training:

```bash
python train.py \
    --data-path ./path/to/TRAIN \
    --epochs 20 \
    --tsne --tsne-samples 500
```

Saves `outputs/tsne_nano_jepa.png`.

---

## Resuming Training

```bash
python train.py \
    --data-path ./path/to/TRAIN \
    --resume outputs/nano_jepa_epoch_009.pth
```

---

## CLI Reference

All hyperparameters can be set via command-line flags. Boolean flags accept `true/false` or bare flags.

### Data

| Flag | Default | Description |
|---|---|---|
| `--data-path` | `./dataset` | Path to ImageFolder root (must contain class subfolders) |
| `--output-dir` | `./outputs` | Where checkpoints and plots are saved |
| `--synthetic` | `false` | Use synthetic data for MFU benchmarking |

### Model

| Flag | Default | Description |
|---|---|---|
| `--img-size` | `64` | Input image resolution |
| `--patch-size` | `8` | ViT patch size |
| `--embed-dim` | `384` | Transformer embedding dimension |
| `--context-depth` | `6` | Context encoder layers |
| `--target-depth` | `6` | Target encoder layers |
| `--predictor-depth` | `3` | Predictor layers |
| `--heads` | `12` | Number of attention heads |
| `--mask-ratio` | `0.75` | Fraction of patches masked |

### Training

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `20` | Number of training epochs |
| `--batch-size` | `256` | Per-GPU batch size |
| `--lr` | `3e-4` | Base learning rate |
| `--scale-lr` | `false` | Scale LR linearly by global batch size / 512 |
| `--weight-decay` | `0.05` | AdamW weight decay |
| `--warmup-epochs` | `2` | Linear warmup epochs |
| `--save-every` | `5` | Save checkpoint every N epochs |
| `--resume` | — | Path to `.pth` checkpoint to resume from |

### Performance

| Flag | Default | Description |
|---|---|---|
| `--compile` | `false` | Enable `torch.compile` on context encoder and predictor |
| `--disable-fp16` | `false` | Disable FP16 mixed precision |
| `--disable-grad-checkpointing` | `false` | Disable gradient checkpointing (faster, more VRAM) |
| `--aug` | `true` | Enable data augmentation (resize, crop, flip) |

### Evaluation

| Flag | Default | Description |
|---|---|---|
| `--tsne` | `false` | Generate t-SNE plot after training |
| `--tsne-samples` | `500` | Number of samples for t-SNE |
| `--benchmark-steps` | `0` | Stop after N steps (for benchmarking) |

---

## MFU Accounting

Model FLOPs Utilization is logged every `--log-interval` steps:

```text
TFLOPS=6.407  MFU=9.857%  theo_TFLOPS=0.161  theo_MFU=0.247%
```

- **TFLOPS / MFU**: Calibrated using `--flops-per-image` (measured FLOPs per forward+backward pass)
- **theo_TFLOPS / theo_MFU**: Theoretical estimate using the `6 × N_params` rule

Defaults are calibrated for T4 GPUs (`--peak-tflops 65.0`). Adjust for your hardware.

---

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or keep gradient checkpointing enabled:

```bash
--batch-size 128
# or remove --disable-grad-checkpointing
```

### `torch.compile` Issues

Try a different compile mode or disable:

```bash
--compile-mode default
# or
--compile false
```

### DDP Hangs

If multi-GPU training hangs, try disabling static graph:

```bash
--static-graph false
```

Enable NCCL debug logging:

```bash
NCCL_DEBUG=INFO python -m torch.distributed.run ...
```

---

## Design Principles

- **Minimal** — No heavy experiment framework, no YAML configs, no package installation step
- **Modular** — Clean separation: config, model, data, training
- **Transparent** — Every hyperparameter is a CLI flag with a sensible default
- **Reproducible** — Seeded RNG, deterministic masking, checkpoint resume

---

## License

MIT — see [LICENSE](LICENSE) for details.