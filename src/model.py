# SPDX-License-Identifier: MIT
"""1D Masked Autoencoder model components."""

import torch
import torch.nn as nn


def _init_weights_vit(module):
    """ViT-style init: trunc_normal_(std=0.02) for Linear/Embedding, ones/zeros for LayerNorm."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=0.02)


class Patches(nn.Module):
    """Split a 1D signal (B, L) into non-overlapping patches (B, N, P)."""

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        B, L = x.shape
        N = L // self.patch_size
        return x.view(B, N, self.patch_size)


class PatchEncoder(nn.Module):
    """Patch embedding + learnable positional embedding + random masking.

    In pretraining mode, returns (x_keep, mask_indices, keep_indices, ids_restore).
    With downstream=True, returns all patch embeddings unmasked.
    """

    def __init__(self, patch_size, projection_dim, mask_proportion, num_patches,
                 dropout_rate=0.0, downstream=False):
        super().__init__()
        self.patch_size = patch_size
        self.projection_dim = projection_dim
        self.mask_proportion = mask_proportion
        self.downstream = downstream
        self.num_patches = num_patches
        self.num_mask = int(mask_proportion * num_patches)
        assert 0 <= self.num_mask < self.num_patches

        self.projection = nn.Linear(patch_size, projection_dim)
        self.position_embedding = nn.Embedding(num_patches, projection_dim)
        self.pos_drop = nn.Dropout(dropout_rate)

    def get_random_indices(self, batch_size, device):
        noise = torch.rand(batch_size, self.num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        len_keep = self.num_patches - self.num_mask
        keep_indices = ids_shuffle[:, :len_keep]
        mask_indices = ids_shuffle[:, len_keep:]
        return mask_indices, keep_indices, ids_restore

    def forward(self, patches: torch.Tensor):
        device = patches.device
        B, N, P = patches.shape
        assert N == self.num_patches, f"Expected N={self.num_patches}, got {N}"

        patch_embeddings = self.projection(patches)

        positions = torch.arange(N, device=device)
        pos_embed = self.position_embedding(positions).unsqueeze(0).to(dtype=patch_embeddings.dtype)
        patch_embeddings = self.pos_drop(patch_embeddings + pos_embed)

        if self.downstream:
            return patch_embeddings

        mask_indices, keep_indices, ids_restore = self.get_random_indices(B, device)
        batch_idx = torch.arange(B, device=device).unsqueeze(-1)
        x_keep = patch_embeddings[batch_idx, keep_indices]
        return x_keep, mask_indices, keep_indices, ids_restore


class MLP(nn.Module):
    """Linear -> GELU -> Dropout -> Linear -> Dropout."""

    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with separate head_dim and model_dim."""

    def __init__(self, model_dim, num_heads, head_dim, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim

        self.q_proj = nn.Linear(model_dim, self.attn_dim)
        self.k_proj = nn.Linear(model_dim, self.attn_dim)
        self.v_proj = nn.Linear(model_dim, self.attn_dim)
        self.out_proj = nn.Linear(self.attn_dim, model_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, N, self.attn_dim)
        out = self.proj_drop(self.out_proj(out))
        return out


class EncoderBlock(nn.Module):
    """Pre-norm Transformer block: LN -> MHA -> skip, LN -> MLP -> skip."""

    def __init__(self, dim, num_heads, head_dim, dropout_rate, mlp_ratio,
                 layer_norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = MultiHeadAttention(
            model_dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout_rate,
        )
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEEncoder(nn.Module):
    """Stack of EncoderBlocks + final LayerNorm."""

    def __init__(self, dim, num_heads, num_layers, head_dim, dropout_rate,
                 mlp_ratio, layer_norm_eps=1e-6):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(
                dim=dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout_rate=dropout_rate,
                mlp_ratio=mlp_ratio,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return self.final_norm(x)


class DecoderBlock(nn.Module):
    """Same structure as EncoderBlock with decoder dimensions."""

    def __init__(self, dim, num_heads, head_dim, dropout_rate, mlp_ratio,
                 layer_norm_eps=1e-6):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = MultiHeadAttention(
            model_dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout_rate,
        )
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = MLP(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout_rate=dropout_rate,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    """Stack of DecoderBlocks + final LayerNorm + linear projection to patch_size."""

    def __init__(self, dec_dim, num_patches, patch_size, num_layers, num_heads,
                 head_dim, dropout_rate, mlp_ratio, layer_norm_eps=1e-6):
        super().__init__()
        self.num_patches = num_patches
        self.dec_dim = dec_dim
        self.patch_size = patch_size

        self.layers = nn.ModuleList([
            DecoderBlock(
                dim=dec_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                dropout_rate=dropout_rate,
                mlp_ratio=mlp_ratio,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(dec_dim, eps=layer_norm_eps)
        self.out_linear = nn.Linear(dec_dim, patch_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        x = self.out_linear(x)
        return x


class MaskedAutoencoder(nn.Module):
    """Full MAE: patches -> encode kept tokens -> insert mask tokens -> decode -> MSE on masked."""

    def __init__(self, patch_layer, patch_encoder, encoder, decoder,
                 enc_projection_dim, dec_projection_dim, loss_fn=None):
        super().__init__()
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.loss_fn = loss_fn if loss_fn is not None else nn.MSELoss()

        self.decoder_embed = nn.Linear(enc_projection_dim, dec_projection_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_projection_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.decoder_position_embedding = nn.Embedding(
            self.patch_encoder.num_patches, dec_projection_dim
        )

    def reconstruct(self, signals: torch.Tensor):
        """Run a forward pass and return pieces useful for visualization.

        Returns (patches, pred_patches, mask_indices) where:
            patches       (B, N, P)      original patches
            pred_patches  (B, N, P)      decoder output for every patch
            mask_indices  (B, N_mask)    indices of patches that were masked
        """
        patches = self.patch_layer(signals)

        x_keep, mask_indices, keep_indices, ids_restore = self.patch_encoder(patches)

        enc_out = self.encoder(x_keep)
        dec_vis = self.decoder_embed(enc_out)

        B, N, P = patches.shape
        N_mask = mask_indices.shape[1]
        mask_tokens = self.mask_token.expand(B, N_mask, -1)

        x_ = torch.cat([dec_vis, mask_tokens], dim=1)

        D_dec = x_.shape[-1]
        x_ = x_.gather(
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, D_dec)
        )

        positions = torch.arange(N, device=signals.device)
        dec_pos = self.decoder_position_embedding(positions).unsqueeze(0).to(dtype=x_.dtype)
        x_ = x_ + dec_pos

        pred_patches = self.decoder(x_)
        return patches, pred_patches, mask_indices

    def calculate_loss(self, signals: torch.Tensor):
        """Returns (loss, target_masked, pred_masked) where loss is MSE on masked patches only."""
        patches, pred_patches, mask_indices = self.reconstruct(signals)
        P = patches.shape[2]

        idx = mask_indices.unsqueeze(-1).expand(-1, -1, P)
        target_masked = torch.gather(patches, dim=1, index=idx)
        pred_masked = torch.gather(pred_patches, dim=1, index=idx)

        total_loss = self.loss_fn(pred_masked, target_masked)
        return total_loss, target_masked, pred_masked

    def forward(self, signals: torch.Tensor):
        return self.calculate_loss(signals)


class DownstreamClassifier(nn.Module):
    """Pretrained encoder + GAP + linear head.

    Linear probing (freeze_backbone=True): backbone stays in eval mode,
    BatchNorm(affine=False) before the head, head dropout disabled.
    Full fine-tuning (freeze_backbone=False): standard dropout before the head.
    """

    def __init__(self, patch_layer, patch_encoder, encoder, output_dim,
                 enc_projection_dim, head_dropout=0.0, freeze_backbone=False):
        super().__init__()
        self.patch_layer = patch_layer
        self.patch_encoder = patch_encoder
        self.encoder = encoder
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            self.bn = nn.BatchNorm1d(enc_projection_dim, affine=False)

        self.head_drop = nn.Dropout(head_dropout)
        self.head = nn.Linear(enc_projection_dim, output_dim)

    def train(self, mode=True):
        # Keep frozen backbone in eval mode (no dropout, frozen BN running stats).
        super().train(mode)
        if mode and self.freeze_backbone:
            self.patch_encoder.eval()
            self.encoder.eval()
        return self

    def forward(self, x):
        patches = self.patch_layer(x)
        feats = self.patch_encoder(patches)
        feats = self.encoder(feats)
        feats = feats.mean(dim=1)

        if self.freeze_backbone:
            feats = self.bn(feats)

        logits = self.head(self.head_drop(feats))
        return logits


def build_mae_from_config(cfg, device="cpu"):
    """Build a MaskedAutoencoder from a model config dict.

    Required keys: signal_size, patch_size, mask_proportion, dropout,
    encoder.{projection_dim, num_heads, head_dim, layers, mlp_ratio},
    decoder.{...}. num_patches is derived from signal_size // patch_size.
    """
    enc = cfg["encoder"]
    dec = cfg["decoder"]
    num_patches = cfg["signal_size"] // cfg["patch_size"]

    patch_layer = Patches(patch_size=cfg["patch_size"])

    patch_encoder = PatchEncoder(
        patch_size=cfg["patch_size"],
        projection_dim=enc["projection_dim"],
        mask_proportion=cfg["mask_proportion"],
        num_patches=num_patches,
        dropout_rate=cfg["dropout"],
        downstream=False,
    )

    encoder = MAEEncoder(
        dim=enc["projection_dim"],
        num_heads=enc["num_heads"],
        num_layers=enc["layers"],
        head_dim=enc["head_dim"],
        dropout_rate=cfg["dropout"],
        mlp_ratio=enc["mlp_ratio"],
        layer_norm_eps=1e-6,
    )

    decoder = MAEDecoder(
        dec_dim=dec["projection_dim"],
        num_patches=num_patches,
        patch_size=cfg["patch_size"],
        num_layers=dec["layers"],
        num_heads=dec["num_heads"],
        head_dim=dec["head_dim"],
        dropout_rate=cfg["dropout"],
        mlp_ratio=dec["mlp_ratio"],
        layer_norm_eps=1e-6,
    )

    mae_model = MaskedAutoencoder(
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        decoder=decoder,
        enc_projection_dim=enc["projection_dim"],
        dec_projection_dim=dec["projection_dim"],
        loss_fn=nn.MSELoss(),
    )
    mae_model.apply(_init_weights_vit)
    mae_model.to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    dec_params = sum(p.numel() for p in decoder.parameters())
    mae_params = sum(p.numel() for p in mae_model.parameters())
    print(f"Encoder parameters: {enc_params}")
    print(f"Decoder parameters: {dec_params}")
    print(f"MAE total params:   {mae_params}")

    return mae_model


def build_downstream_from_mae(mae_model, output_dim, cfg, device="cpu",
                              finetune_strategy="full", downstream_dropout=0.0,
                              head_dropout=0.0, train_patch_encoder=False):
    """Build a DownstreamClassifier from a pretrained MAE.

    The encoder is rebuilt with downstream-specific dropout and then loaded
    with the pretrained weights, so downstream settings always take effect.
    finetune_strategy is "full" or "linear".
    """
    patch_layer = mae_model.patch_layer
    patch_encoder = mae_model.patch_encoder

    enc = cfg["encoder"]
    encoder = MAEEncoder(
        dim=enc["projection_dim"],
        num_heads=enc["num_heads"],
        num_layers=enc["layers"],
        head_dim=enc["head_dim"],
        dropout_rate=downstream_dropout,
        mlp_ratio=enc["mlp_ratio"],
        layer_norm_eps=1e-6,
    ).to(device)
    encoder.load_state_dict(mae_model.encoder.state_dict())

    patch_encoder.downstream = True

    freeze_backbone = (finetune_strategy == "linear")

    if freeze_backbone and head_dropout > 0.0:
        head_dropout = 0.0

    downstream_model = DownstreamClassifier(
        patch_layer=patch_layer,
        patch_encoder=patch_encoder,
        encoder=encoder,
        output_dim=output_dim,
        enc_projection_dim=enc["projection_dim"],
        head_dropout=head_dropout,
        freeze_backbone=freeze_backbone,
    )
    downstream_model.head.apply(_init_weights_vit)
    downstream_model.to(device)

    if finetune_strategy == "full":
        for p in patch_encoder.parameters():
            p.requires_grad = train_patch_encoder
        for p in encoder.parameters():
            p.requires_grad = True
        for p in downstream_model.head.parameters():
            p.requires_grad = True
    elif finetune_strategy == "linear":
        for p in patch_encoder.parameters():
            p.requires_grad = False
        for p in encoder.parameters():
            p.requires_grad = False
        for p in downstream_model.head.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown finetune_strategy: {finetune_strategy}")

    return downstream_model
