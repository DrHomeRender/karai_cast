

import torch
import math
import torch.nn as nn
from .ScaledDotProductAttention import *

# 여러개의 ScaledDotProductAttention 을 병렬로 수행하고 concat+projection 하는구조 머리를 여러개 둬서 다양한 관점을 보게하는 구조
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = self.d_v = d_model // n_head
        self.n_head = n_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # (batch, head, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask)

        # (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)
