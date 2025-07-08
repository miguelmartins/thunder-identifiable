import math

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath, trunc_normal_


class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        nb_classes: int,
    ) -> None:
        """
        Initializing ClassificationHead.
        :param embed_dim: dimension of the input embedding.
        :param nb_classes: number of classes.
        """
        super().__init__()
        self.linear = nn.Linear(embed_dim, nb_classes)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param src: input embedding.
        :return classification prediction.
        """
        return self.linear(src)


class GridSearchClassificationHead(nn.Module):
    def __init__(
        self,
        nb_heads: int,
        embed_dim: int,
        nb_classes: int,
    ) -> None:
        """
        Initializing GridSearchClassificationHead.
        :param nb_heads: number of classification heads to train.
        :param embed_dim: dimension of the input embedding.
        :param nb_classes: number of classes.
        """
        super().__init__()
        self.nb_heads = nb_heads
        self.heads = nn.ModuleList(
            [ClassificationHead(embed_dim, nb_classes) for _ in range(self.nb_heads)]
        )

    def forward(self, src: torch.Tensor | list) -> torch.Tensor:
        """
        Forward pass.
        :param src: input embedding.
        :return classification prediction per head.
        """
        if type(src) == torch.Tensor:
            out = [self.heads[i](src) for i in range(self.nb_heads)]
        else:
            assert type(src) == list
            out = [self.heads[i](src[i]) for i in range(self.nb_heads)]
        return out


# Adapted from https://github.com/rstrudel/segmenter/tree/master
def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls: int,
        d_encoder: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        drop_path_rate: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.d_encoder = d_encoder

        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model**-0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x):
        GS = int(math.sqrt(x.shape[1]))

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


class GridSearchMaskTransformer(nn.Module):
    def __init__(
        self,
        nb_heads: int,
        n_cls: int,
        d_encoder: int,
        n_layers: int,
        n_heads: int,
        d_model: int,
        d_ff: int,
        drop_path_rate: float,
        dropout: float,
    ) -> None:
        """
        Initializing GridSearchMaskTransformer.
        :param nb_heads: number of segmentation heads to train.
        :param n_cls:
        :param d_encoder:
        :param n_layers:
        :param n_heads:
        :param d_model:
        :param d_ff:
        :param drop_path_rate:
        :param dropout:
        """
        super().__init__()
        self.nb_heads = nb_heads
        self.heads = nn.ModuleList(
            [
                MaskTransformer(
                    n_cls,
                    d_encoder,
                    n_layers,
                    n_heads,
                    d_model,
                    d_ff,
                    drop_path_rate,
                    dropout,
                )
                for _ in range(self.nb_heads)
            ]
        )

    def forward(self, src: torch.Tensor | list) -> torch.Tensor:
        """
        Forward pass.
        :param src: input embedding.
        :return segmentation prediction per head.
        """
        if type(src) == torch.Tensor:
            out = [self.heads[i](src) for i in range(self.nb_heads)]
        else:
            assert type(src) == list
            out = [self.heads[i](src[i]) for i in range(self.nb_heads)]
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim**-0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
