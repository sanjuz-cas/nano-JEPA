from argparse import ArgumentParser
from dataclasses import dataclass, fields


def str2bool(v):
    if isinstance(v, bool):
        return v

    if isinstance(v, str):
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        if v.lower() in ("no", "false", "f", "n", "0"):
            return False

    raise ValueError(f"Cannot convert {v} to bool")


@dataclass
class Config:
    # Data
    data_path: str = "./dataset"
    output_dir: str = "./outputs"

    synthetic: bool = False
    synthetic_size: int = 100000

    # Model
    img_size: int = 64
    patch_size: int = 8
    embed_dim: int = 384
    context_depth: int = 6
    target_depth: int = 6
    predictor_depth: int = 3
    heads: int = 12
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75

    # Training
    epochs: int = 20
    batch_size: int = 256  # per-GPU batch size
    accumulation_steps: int = 1  # Gradient accumulation steps
    lr: float = 3e-4
    scale_lr: bool = False
    lr_scale_base_batch: int = 512
    weight_decay: float = 0.05
    warmup_epochs: int = 2

    # EMA target encoder
    momentum: float = 0.996
    momentum_end: float = 1.0

    # Loss regularization
    std_weight: float = 0.05
    cov_weight: float = 0.04
    disable_vicreg: bool = False

    # Optimization
    clip_grad: float = 0.0
    workers: int = 4
    prefetch: int = 2  
    log_interval: int = 20
    save_every: int = 5
    resume: str = ""

    # Performance
    compile: bool = False
    # "reduce-overhead" uses CUDA Graphs, massive MFU boost for static ViT shapes
    compile_mode: str = "reduce-overhead" 
    disable_grad_checkpointing: bool = False
    static_graph: bool = True
    aug: bool = True
    disable_fp16: bool = False
    
    # 0.0 triggers dynamic calculation/auto-detection in train.py
    flops_per_image: float = 0.0 
    peak_tflops: float = 0.0     

    # Benchmark / eval
    benchmark_steps: int = 0
    tsne: bool = False
    tsne_samples: int = 500

    seed: int = 42


def parse_config() -> Config:
    parser = ArgumentParser(description="Nano JEPA Config")

    for field in fields(Config):
        default = getattr(Config, field.name)
        name = "--" + field.name.replace("_", "-")

        if isinstance(default, bool):
            parser.add_argument(
                name,
                type=str2bool,
                nargs="?",
                const=True,
                default=default,
                metavar="true/false",
            )
        else:
            parser.add_argument(
                name,
                type=type(default),
                default=default,
            )

    # Compatibility with older launchers that pass --local-rank.
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)

    args, _ = parser.parse_known_args()

    valid_fields = {f.name for f in fields(Config)}
    kwargs = {k: v for k, v in vars(args).items() if k in valid_fields}

    return Config(**kwargs)
