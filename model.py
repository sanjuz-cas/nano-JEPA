import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        assert dim % heads == 0

        self.heads = heads
        self.head_dim = dim // heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        mlp_ratio: float,
        grad_ckpt: bool = False,
    ):
        super().__init__()

        self.grad_ckpt = grad_ckpt
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)

        return self.norm(x)


def _init_weights(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class NanoJEPA(nn.Module):
    """
    Nano JEPA matching your reported configuration:

    - Input: 64x64
    - Patch: 8x8
    - Patches: 64
    - Embed dim: 384
    - Context encoder: 6 layers
    - Target encoder: 6 layers
    - Predictor: 3 layers
    - Heads: 12
    - Mask ratio: 75%
    - EMA target encoder
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        embed_dim: int = 384,
        context_depth: int = 6,
        target_depth: int = 6,
        predictor_depth: int = 3,
        heads: int = 12,
        mlp_ratio: float = 4.0,
        mask_ratio: float = 0.75,
        grad_ckpt: bool = True,
    ):
        super().__init__()

        assert img_size % patch_size == 0

        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.num_patches = (img_size // patch_size) ** 2
        self.num_keep = int(round(self.num_patches * (1.0 - mask_ratio)))

        # Online/student branch
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.context_encoder = Transformer(
            dim=embed_dim,
            depth=context_depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            grad_ckpt=grad_ckpt,
        )

        # Target/teacher branch: separate parameters, EMA-updated, no gradients.
        self.target_patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.target_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        self.target_encoder = Transformer(
            dim=embed_dim,
            depth=target_depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            grad_ckpt=False,
        )

        # Predictor
        self.predictor = Transformer(
            dim=embed_dim,
            depth=predictor_depth,
            heads=heads,
            mlp_ratio=mlp_ratio,
            grad_ckpt=grad_ckpt,
        )

        # Small head helps match the reported ~26.96M parameter count.
        self.predictor_head = nn.Linear(embed_dim, embed_dim)

        self.apply(_init_weights)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.target_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.init_target()

        # Freeze target branch.
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        for p in self.target_patch_embed.parameters():
            p.requires_grad = False

        self.target_pos_embed.requires_grad = False

    @torch.no_grad()
    def init_target(self):
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        self.target_patch_embed.load_state_dict(self.patch_embed.state_dict())
        self.target_pos_embed.copy_(self.pos_embed)

    @torch.no_grad()
    def update_target(self, momentum: float):
        # target = momentum * target + (1 - momentum) * online
        alpha = 1.0 - momentum

        for pt, po in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            pt.lerp_(po, alpha)

        for pt, po in zip(self.target_patch_embed.parameters(), self.patch_embed.parameters()):
            pt.lerp_(po, alpha)

        self.target_pos_embed.lerp_(self.pos_embed, alpha)

    def make_mask_ids(self, B: int, device: torch.device):
        noise = torch.rand(B, self.num_patches, device=device)
        ids = torch.argsort(noise, dim=1)

        visible_ids = ids[:, : self.num_keep]
        mask_ids = ids[:, self.num_keep :]

        return visible_ids, mask_ids

    def forward(self, x):
        B = x.shape[0]
        device = x.device

        # Online visible tokens.
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed

        # Target full-image representations, no gradients.
        with torch.no_grad():
            target_tokens = self.target_patch_embed(x).flatten(2).transpose(1, 2)
            target_tokens = target_tokens + self.target_pos_embed
            target = self.target_encoder(target_tokens)

        visible_ids, mask_ids = self.make_mask_ids(B, device)

        visible_idx = visible_ids.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        mask_idx = mask_ids.unsqueeze(-1).expand(-1, -1, self.embed_dim)

        visible_tokens = torch.gather(tokens, dim=1, index=visible_idx)
        context_out = self.context_encoder(visible_tokens)

        # Predictor input:
        #   - mask tokens at all positions
        #   - replace visible positions with context encoder outputs
        pred_input = (
            self.mask_token + self.pos_embed
        ).expand(B, self.num_patches, self.embed_dim).contiguous()

        pred_input.scatter_(dim=1, index=visible_idx, src=context_out)

        pred = self.predictor(pred_input)
        pred = self.predictor_head(pred)

        pred_mask = torch.gather(pred, dim=1, index=mask_idx)
        target_mask = torch.gather(target, dim=1, index=mask_idx).detach()

        # Flatten to (B * num_masked_patches, C)
        pred_mask = pred_mask.reshape(-1, self.embed_dim)
        target_mask = target_mask.reshape(-1, self.embed_dim)

        return pred_mask, target_mask


def variance_loss(x: torch.Tensor) -> torch.Tensor:
    std = torch.sqrt(x.var(dim=0) + 1e-4)
    return F.relu(1.0 - std).mean()


def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    B, C = x.shape

    if B < 2:
        return x.new_zeros(())

    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (B - 1)

    mask = ~torch.eye(C, dtype=torch.bool, device=x.device)
    off_diag = cov[mask]

    return off_diag.pow(2).sum() / C


def jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    std_weight: float = 0.05,
    cov_weight: float = 0.04,
) -> torch.Tensor:
    pred = pred.float()
    target = target.float()

    loss = F.smooth_l1_loss(pred, target, beta=1.0)

    if std_weight > 0:
        loss = loss + std_weight * variance_loss(pred)

    if cov_weight > 0:
        loss = loss + cov_weight * covariance_loss(pred)

    return loss