
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from ..functions.quadtree_attention import score_computation_op, value_aggregation_op
from einops.einops import rearrange


class QTAttA(nn.Module):
    def __init__(
        self,
        nhead,
        dim,
        topks=[32, 32, 32, 32],
        scale=None,
        use_dropout=False,
        attention_dropout=0.1,
    ):
        super().__init__()
        self.use_dropout = use_dropout
        self.topks = topks
        self.nhead = nhead
        self.dim = dim

    def process_coarse_level(self, query, key, value, topk):
        bs, c, h, w = key.shape
        cur_dim = key.shape[1] // self.nhead

        key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]
        query = rearrange(query, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)

        QK = torch.einsum("nlhd,nshd->nlsh", query, key)
        softmax_temp = 1.0 / cur_dim ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)

        # mask out top K tokens
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)
        mask = torch.ones_like(A)
        mask = mask.scatter(dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float())

        # message is only computed within the unmasked
        message = torch.einsum("nlsh,nshd->nlhd", A * mask, value)  # .reshape(bs, h, w, self.nhead, cur_dim)

        return A, message, topk_score, topk_idx

    def process_fine_level(self, query, key, value, topk_score, topk_pos, topk_prev, topk, final=False):
        bs, c, h, w = key.shape

        cur_dim = key.shape[1] // self.nhead
        key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]
        value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, cur_dim)  # [N, S, H, D]

        query = query.view(bs, c, h // 2, 2, w // 2, 2)
        query = rearrange(query, "b c h t1 w t2-> b (h w) (t1 t2) c ").view(bs, -1, 4, self.nhead, cur_dim)

        # convert 2d coordinates to 1d index
        idx_gather = []
        topk_pos = topk_pos * 2
        for x in [0, 1]:
            for y in [0, 1]:
                idx = (topk_pos[0] + x) * w + topk_pos[1] + y  # convert to index
                idx_gather.append(idx)

        idx = torch.stack(idx_gather, dim=3)  # [N, L, K, 4, H, D]

        # Compute score
        # query: [b, N, 4, H, D]
        # key: [b, 4N, H, D]
        # idx: [b, N, K, 4, H]
        # QK: [b, N, 4, 4K, H]
        QK = score_computation_op(query, key.contiguous(), idx.view(bs, -1, topk_prev * 4, self.nhead))
        QK = rearrange(QK, "n l w (k f) h -> n l w k f h", k=topk_prev, f=4)
        softmax_temp = 1.0 / cur_dim ** 0.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=-2)  # [N, L//scale**i, K, 4, H]
        # Score redistribution
        topk_score = topk_score.unsqueeze(-2).unsqueeze(2)
        A = (A * topk_score).reshape(bs, -1, 4, topk_prev * 4, self.nhead)
        idx = idx.view(bs, -1, 1, topk_prev * 4, self.nhead).repeat(1, 1, 4, 1, 1)  # [N, L,4, K*4, H]
        topk_score, topk_idx = torch.topk(A, dim=-2, k=topk, largest=True)

        if not final:
            mask = torch.ones_like(A)
            mask = mask.scatter(dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float())
            message = value_aggregation_op(A * mask, value.contiguous(), idx)
        else:
            message = value_aggregation_op(A, value.contiguous(), idx)

        if not final:
            topk_idx = torch.gather(idx, index=topk_idx, dim=-2)
            topk_idx = rearrange(topk_idx, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2)  # reshape back
            topk_score = rearrange(
                topk_score, "b (h w) (t1 t2) k nh -> b (h t1 w t2) k nh", h=h // 2, t1=2
            )  # reshape back

        return A, message, topk_score, topk_idx

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Multi-head quadtree attention
        Args:
            queries: Query pyramid [N, C, H, W]
            keys: Key pyramid [N, C, H, W]
            values: Value pyramid [N, C, H, W]
        Returns:
            message: (N, C, H, W)
        """