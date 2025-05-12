import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_


# ──────────────────────────────────────────────────────────────────────────────
#  CUSTOM ACTIVATION
# ──────────────────────────────────────────────────────────────────────────────
class RationalActivation(nn.Module):
    """Learnable rational activation P3(x) / (1 + |Q3(x)|)."""

    def __init__(self):
        super().__init__()
        # Numerator coefficients [p0, p1, p2, p3]
        self.p = nn.Parameter(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        # Denominator coefficients [q0, q1, q2]
        self.q = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))

    def forward(self, x):
        # Compute numerator and denominator
        num = self.p[0] + self.p[1] * x + self.p[2] * x**2 + self.p[3] * x**3
        den = 1 + torch.abs(self.q[0] * x + self.q[1] * x**2 + self.q[2] * x**3)
        return num / den


# ──────────────────────────────────────────────────────────────────────────────
#  GROUP KAN LAYER
# ──────────────────────────────────────────────────────────────────────────────
class GroupKANLayer(nn.Module):
    """Splits features into groups, applies Linear + RationalActivation per group."""

    def __init__(self, in_features, out_features, num_groups=8):
        super().__init__()
        assert in_features % num_groups == 0 and out_features % num_groups == 0
        # group size per chunk
        gin = in_features // num_groups
        gout = out_features // num_groups
        # one Linear + activation per group
        self.linears = nn.ModuleList([nn.Linear(gin, gout) for _ in range(num_groups)])
        self.activs = nn.ModuleList([RationalActivation() for _ in range(num_groups)])

    def forward(self, x):
        # Split channels into num_groups chunks
        chunks = torch.chunk(x, len(self.linears), dim=-1)
        # Apply each linear + activation, then concatenate
        outs = [
            act(linear(c)) for c, linear, act in zip(chunks, self.linears, self.activs)
        ]
        return torch.cat(outs, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
#  MAIN MODEL: KATForAMP
# ──────────────────────────────────────────────────────────────────────────────
class KATForAMP(VisionTransformer):
    """
    Vision Transformer backbone treating a peptide as 1D patches.
    Replaces standard MLP with grouped KAN layers.
    Returns both logits and token embeddings for per-sequence pooling.
    """

    def __init__(
        self,
        seq_length=3000,
        embed_dim=768,
        num_classes=1,
        neurons=32,  # unused but kept for signature
        activation="relu",
        dropout_rate=0.5,
        **vit_kwargs,
    ):
        # Initialize ViT with 1×1 patches over length = seq_length
        super().__init__(
            img_size=seq_length,
            patch_size=1,
            in_chans=1,
            embed_dim=embed_dim,
            num_classes=num_classes,
            **vit_kwargs,
        )
        # Token embedding for amino acids (0–25)
        self.embedding = nn.Embedding(26, embed_dim)
        # CLS token & positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_length + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        # Replace MLP in each block with GroupKAN
        self.blocks = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": blk.attn,
                        "norm1": blk.norm1,
                        "kan": GroupKANLayer(embed_dim, embed_dim),
                        "norm2": blk.norm2,
                    }
                )
                for blk in self.blocks
            ]
        )
        # Classification head and dropout
        self.head = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        # Activation lookup (unused in current forward)
        self.act = self._get_act(activation)
        # Initialize weights
        self.apply(self._init_weights)

    def _get_act(self, name):
        table = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "softplus": nn.Softplus(),
            "softsign": nn.Softsign(),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
            "elu": nn.ELU(),
            "exponential": nn.Identity(),  # linear placeholder
        }
        if name not in table:
            raise ValueError(f"Unknown activation {name}")
        return table[name]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)

    def forward_features(self, x_ids):
        """
        Compute embeddings for CLS and tokens.
        Returns:
          cls_emb  : [B, embed_dim]
          tok_embs : [B, L, embed_dim]
        """
        # Embed tokens & prepend CLS
        x = self.embedding(x_ids)
        cls_t = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_t, x), dim=1)
        # Add positional embedding
        x = x + self.pos_embed[:, : x.size(1), :]
        # Transformer blocks with custom KAN MLP
        for blk in self.blocks:
            x = x + blk["attn"](blk["norm1"](x))
            x = x + blk["kan"](blk["norm2"](x))
        x = self.norm(x)
        # Split out CLS vs token embeddings
        cls_emb = x[:, 0]
        tok_embs = x[:, 1:]
        return cls_emb, tok_embs

    def forward(self, x_ids):
        """Return logits and token embeddings for pooling."""
        cls_emb, tok_embs = self.forward_features(x_ids)
        logits = self.head(self.dropout(cls_emb)).squeeze(-1)
        return logits, tok_embs


# ──────────────────────────────────────────────────────────────────────────────
#  MODEL FACTORY
# ──────────────────────────────────────────────────────────────────────────────
def create_kat_amp_model(**kwargs):
    """Instantiate KATForAMP; drop training-only args, enable DataParallel."""
    # Remove keys that belong to training script
    for key in ("optimizer", "learning_rate", "kernelOne"):
        kwargs.pop(key, None)
    model = KATForAMP(**kwargs)
    # Wrap in DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model
