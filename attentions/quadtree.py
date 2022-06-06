
"""
    Paper: QuadTree Attention for Vision Transformers
    Link: https://arxiv.org/abs/2201.02767
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import sys
sys.path.append("./QuadTreeAttention")
from QuadtreeAttention.modules.quadtree_attention import QTAttA, QTAttB
from einops.einops import rearrange

class QTAttAPytorch(nn.Module):
    def __init__(self, nhead, dim, scale, use_dropout=False, attention_dropout=0.1, topk=1):
        super().__init__()
        self.use_dropout = use_dropout
        self.topk = topk
        self.nhead = nhead
        self.dim = dim

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """Quadtree attention
        Args:
            query: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        bs = queries[0].shape[0]
        # Compute the unnormalized attention and apply the masks
        message = 0
        for i, (query, key, value) in enumerate(zip(reversed(queries), reversed(keys), reversed(values))):
            bs, c, h, w = key.shape
            key = rearrange(key, "b c h w -> b (h w) c").view(bs, -1, self.nhead, self.dim)  # [N, S, H, D]
            value = rearrange(value, "b c h w -> b (h w) c").view(bs, -1, self.nhead, self.dim)  # [N, S, H, D]
            value_old = value
            if i == 0:
                query = rearrange(query, "b c h w -> b (h w) c").view(bs, -1, self.nhead, self.dim)
                QK = torch.einsum("nlhd,nshd->nlsh", query, key)
            else:
                query = query.view(bs, c, h // 2, 2, w // 2, 2)
                query = rearrange(query, "b c h t1 w t2-> b (h w) (t1 t2) c ").view(bs, -1, 4, self.nhead, self.dim)
                topk_pos = topk_pos * 2
                key_gather = []
                value_gather = []
                idx_gather = []

                for x in [0, 1]:
                    for y in [0, 1]:
                        idx = (topk_pos[0] + x) * w + topk_pos[1] + y  # convert to index

                        idx_gather.append(idx)
                        idx = idx.view(bs, -1, self.nhead, 1).repeat(1, 1, 1, self.dim)  # [N, L, K, H, D]

                        k = torch.gather(key, index=idx, dim=1).view(bs, -1, self.topk, self.nhead, self.dim)

                        v = torch.gather(value, index=idx, dim=1).view(bs, -1, self.topk, self.nhead, self.dim)
                        key_gather.append(k)
                        value_gather.append(v)
                idx = torch.stack(idx_gather, dim=3)  # [N, L, K, 4, H, D]

                # query: [b, N, 4, H, D]
                # key: [b, 4N, H, D]
                # idx: [b, N, 4K, H, D]
                # QK: [b, N, 4, K, 4, H]

                key_gather = torch.stack(key_gather, dim=3)  # [N, L, K, 4, H, D]
                value = torch.stack(value_gather, dim=3)  # [N, L, K, 4, H, D]

                QK = torch.einsum("nlwhd,nlkfhd->nlwkfh", query, key_gather)

            softmax_temp = 1.0 / self.dim ** 0.5  # sqrt(D)
            A = torch.softmax(softmax_temp * QK, dim=-2)  # [N, L//scale**i, K, 4, H]

            if i != 0:
                # A: [b, N, 4, K, 4, H]
                # topk_score: [b, N, 1, K, 1, H]
                # return: A, [b, N, 4, K*4, H]
                A = (A * topk_score.unsqueeze(-2).unsqueeze(2)).reshape(bs, -1, 4, self.topk * 4, self.nhead)
                # value: [b, N, K, 4, H, D]
                value = value.reshape(bs, -1, self.topk * 4, self.nhead, self.dim)  # [N, L, K*4, H, D]

            topk_score, topk_idx = torch.topk(A, dim=-2, k=self.topk, largest=True)  # [N, L, 4, K, H]

            mask = torch.ones_like(A)
            if i != len(keys) - 1:
                mask = mask.scatter(dim=-2, index=topk_idx, src=torch.zeros_like(topk_idx).float())
            if i == 0:
                message += torch.einsum("nlsh,nshd->nlhd", A * mask, value)
                # [b, N, H, D]
            else:
                # A: [b, N, 4, 4K, H]
                # value: [b, N, 4K, H, D]
                # message: [b, N, 4, H, D]
                new_message = torch.einsum("nlwkh,nlkhd->nlwhd", A * mask, value)

                idx = idx.view(bs, -1, 1, self.topk * 4, self.nhead).repeat(1, 1, 4, 1, 1)  # [N, L,4, K*4, H]
                # A: [b, N, 4, K*4, H]
                # value_old: [b, 4N, H, D]
                # idx: [b, N, 4, 4K, H]

                message = message.unsqueeze(2) + new_message
                message = message.view(bs, h // 2, w // 2, 2, 2, self.nhead, self.dim)
                message = rearrange(message, "b h w t1 t2 nh d -> b (h t1 w t2) nh d")  # reshape

                topk_idx = torch.gather(idx, index=topk_idx, dim=-2)

                topk_idx = topk_idx.view(bs, h // 2, w // 2, 2, 2, self.topk, self.nhead)