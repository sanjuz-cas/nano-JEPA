# Nano JEPA

Minimal, high-performance PyTorch trainer for **Nano JEPA** on **2× NVIDIA T4 GPUs**.

This repository is designed for:

- Self-supervised JEPA-style pretraining
- Kaggle multi-GPU training
- High MFU / throughput benchmarking
- Clean, minimal, engineer-friendly code structure

---

## Model Specification

This implementation matches the Nano JEPA configuration:

| Component | Value |
|---|---:|
| Total Parameters | ~26.96M |
| Input Resolution | 64 × 64 |
| Patch Size | 8 × 8 |
| Total Patches | 64 |
| Embedding Dimension | 384 |
| Context Encoder Layers | 6 |
| Target Encoder Layers | 6 |
| Predictor Layers | 3 |
| Attention Heads | 12 |
| Mask Ratio | 75% |
| Optimizer | AdamW |
| Base LR | 3e-4 |
| Precision | FP16 mixed precision |
| Gradient Checkpointing | Enabled by default |

---

## Reference Performance

Target/reference numbers from the original Nano JEPA HPC analysis:

| Metric | Phase 1: 1× T4 | Phase 2: 2× T4 | Phase 3: 2× T4 |
|---|---:|---:|---:|
| Global Batch Size | 512 | 512 | 2048 |
| Per-GPU Batch Size | 512 | 256 | 1024 |
| Peak Throughput | 985 img/sec | 1877 img/sec | ~22.7 TFLOPS/s |
| MFU | 0.035% | 0.035% | ~25% projected |
| VRAM / GPU | ~922 MB | ~922 MB | ~3.6 GB |

DDP efficiency for Phase 2:

```text
1877 / (2 × 985) × 100% ≈ 95.2%
```

T4 roofline reference:

| Metric | Value |
|---|---:|
| T4 Peak FP16 Compute | 65 TFLOPS |
| T4 Memory Bandwidth | 320 GB/s |
| Required Intensity for 100% MFU | 203 FLOPs/Byte |
| Actual Model Intensity | ~4.2 FLOPs/Byte |

The workload is memory-bandwidth bound at small batch sizes. Increasing the global batch size to 2048 increases arithmetic intensity and improves MFU substantially.

---

## Repository Structure

```text
nano-jepa/
├── README.md
├── .gitignore
├── requirements.txt
├── config.py
├── model.py
├── data.py
├── train.py
└── scripts/
    └── run_kaggle.sh
```

### File Responsibilities

| File | Purpose |
|---|---|
| `config.py` | Dataclass-based configuration and CLI parser |
| `model.py` | Nano JEPA model, EMA target encoder, JEPA loss |
| `data.py` | CIFAR-10 dataset, synthetic benchmark dataset, DataLoaders |
| `train.py` | DDP training loop, MFU logging, checkpointing, t-SNE export |
| `scripts/run_kaggle.sh` | Convenience launcher for Kaggle 2× T4 runs |

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/nano-jepa.git
cd nano-jepa
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For t-SNE export, the following packages are required:

```bash
pip install scikit-learn matplotlib
```

They are already included in `requirements.txt`, but can be removed if not needed.

---

## Quickstart: Kaggle 2× T4

### Phase 2: 2 GPUs, global batch 512

```bash
OMP_NUM_THREADS=4 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --batch-size 256 \
    --epochs 20 \
    --lr 3e-4 \
    --workers 4 \
    --log-interval 10
```

Global batch size:

```text
2 × 256 = 512
```

---

### Phase 3: 2 GPUs, global batch 2048

```bash
OMP_NUM_THREADS=4 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --batch-size 1024 \
    --epochs 20 \
    --lr 3e-4 \
    --scale-lr \
    --warmup-epochs 2 \
    --workers 4 \
    --log-interval 10 \
    --compile
```

Global batch size:

```text
2 × 1024 = 2048
```

Effective LR:

```text
3e-4 × 2048 / 512 = 1.2e-3
```

---

### Pure MFU benchmark

Use synthetic data to remove CIFAR augmentation and disk overhead:

```bash
OMP_NUM_THREADS=4 python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --synthetic \
    --batch-size 1024 \
    --benchmark-steps 100 \
    --aug false \
    --disable-vicreg \
    --compile \
    --disable-grad-checkpointing \
    --log-interval 10
```

If you run out of VRAM, remove:

```bash
--disable-grad-checkpointing
```

or reduce batch size:

```bash
--batch-size 768
```

or:

```bash
--batch-size 512
```

---

## Using the Helper Script

Make the script executable:

```bash
chmod +x scripts/run_kaggle.sh
```

Run Phase 2:

```bash
./scripts/run_kaggle.sh phase2
```

Run Phase 3:

```bash
./scripts/run_kaggle.sh phase3
```

Run MFU benchmark:

```bash
./scripts/run_kaggle.sh bench
```

Run training with t-SNE export:

```bash
./scripts/run_kaggle.sh tsne
```

---

## Single-GPU Usage

For 1× GPU:

```bash
python train.py \
    --batch-size 512 \
    --epochs 20 \
    --lr 3e-4
```

For a quick single-GPU benchmark:

```bash
python train.py \
    --synthetic \
    --batch-size 512 \
    --benchmark-steps 100 \
    --aug false \
    --disable-vicreg
```

---

## CLI Flags

Boolean flags accept either bare flags or explicit `true/false`.

Examples:

```bash
--compile
--compile true
--compile false
```

Important flags:

| Flag | Default | Meaning |
|---|---:|---|
| `--batch-size` | `256` | Per-GPU batch size |
| `--epochs` | `20` | Number of training epochs |
| `--lr` | `3e-4` | Base learning rate |
| `--scale-lr` | `false` | Scale LR by global batch size / 512 |
| `--compile` | `false` | Use `torch.compile` on context encoder and predictor |
| `--disable-grad-checkpointing` | `false` | Disable gradient checkpointing for higher throughput |
| `--synthetic` | `false` | Use synthetic data for benchmarking |
| `--benchmark-steps` | `0` | Stop after N global steps |
| `--aug` | `true` | Enable data augmentation |
| `--fp16` | `true` | Enable FP16 mixed precision |
| `--disable-vicreg` | `false` | Disable variance/covariance regularization |
| `--tsne` | `false` | Export t-SNE after training |
| `--flops-per-image` | `23.1e6` | FLOPs/image used for MFU logging |
| `--peak-tflops` | `65.0` | T4 peak FP16 TFLOPS |

---

## MFU Accounting

The default MFU accounting uses:

```bash
--flops-per-image 23.1e6
--peak-tflops 65.0
```

This is calibrated to match the reported Phase-1 number:

```text
985 img/sec × 23.1 MFLOPs/img ≈ 22.75 GFLOPS
22.75 GFLOPS / 65 TFLOPS ≈ 0.035%
```

If you want a theoretical transformer-style estimate instead:

```bash
--flops-per-image 161.8e6
```

because:

```text
6 × 26.96M ≈ 161.8 MFLOPs/image
```

Note that this will not match the MFU values from the original performance sheet.

---

## Checkpointing

Checkpoints are saved to:

```text
outputs/
```

Example:

```text
outputs/nano_jepa_epoch_019.pth
```

Resume training:

```bash
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --resume outputs/nano_jepa_epoch_019.pth
```

---

## t-SNE Export

Run:

```bash
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --batch-size 1024 \
    --scale-lr \
    --compile \
    --tsne
```

This saves:

```text
outputs/tsne_nano_jepa.png
```

---

## Distributed Implementation Notes

This trainer follows PyTorch distributed best practices:

### 1. `@record`

`train.py` decorates `main()` with:

```python
@record
def main():
    ...
```

This allows TorchElastic to propagate the root-cause worker error to the launcher.

When launched with `torch.distributed.run`, uncaught exceptions are written to the file specified by:

```text
TORCHELASTIC_ERROR_FILE
```

This makes multi-GPU crashes easier to debug.

---

### 2. `destroy_process_group()`

`train.py` calls:

```python
dist.destroy_process_group()
```

near the end of training.

This is important because failing to destroy the process group can cause NCCL hangs on exit, especially when multiple process groups exist.

---

### 3. NCCL backend

This trainer uses NCCL for CUDA GPU training:

```python
dist.init_process_group(backend="nccl")
```

This is the recommended backend for distributed GPU training.

---

## Debugging

### Enable detailed distributed debug logs

```bash
TORCH_CPP_LOG_LEVEL=INFO \
TORCH_DISTRIBUTED_DEBUG=DETAIL \
OMP_NUM_THREADS=4 \
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --batch-size 256
```

`TORCH_DISTRIBUTED_DEBUG=DETAIL` can help detect collective mismatches and rank desynchronization.

---

### NCCL debug logs

```bash
NCCL_DEBUG=INFO \
python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py \
    --batch-size 256
```

---

### Common warnings

These warnings are usually harmless on Kaggle:

```text
Setting OMP_NUM_THREADS environment variable for each process to be 1
```

Fix by explicitly setting:

```bash
OMP_NUM_THREADS=4
```

Another harmless warning:

```text
The hostname of the client socket cannot be retrieved. err=-3
```

This does not affect training.

---

## Troubleshooting

### `UnboundLocalError: cannot access local variable 'torch'`

This happens if `import torch._dynamo` is placed inside `main()`.

This repository avoids that by importing `torch._dynamo` at the top level.

---

### OOM

Reduce batch size:

```bash
--batch-size 768
```

or:

```bash
--batch-size 512
```

or keep gradient checkpointing enabled by removing:

```bash
--disable-grad-checkpointing
```

---

### `torch.compile` issues

Remove:

```bash
--compile
```

or try:

```bash
--compile-mode default
```

or:

```bash
--compile-mode reduce-overhead
```

If DDP with `static_graph` causes issues:

```bash
--static-graph false
```

---

## Design Principles

This repo intentionally stays minimal:

- No heavy experiment framework
- No YAML dependency
- No package installation step required
- One config dataclass
- One model file
- One data file
- One trainer file

The goal is to make the training loop transparent and easy to profile.

---

## License

MIT.