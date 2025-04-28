
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MaskedScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # mask == 0인 곳은 -1e9로 가려줌 (softmax에서 0으로 만들기)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output, attn

def generateSquareSubsequentMask(seq_len):
    """
    미래 시점을 보지 못하게 가리는 look-ahead mask
    shape: (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # False → 가려짐 (0), True → 통과 (1)


# (단일 입력 시퀀스) → Encoder → Flatten → Regression

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = self.d_v = d_model // n_head
        self.n_head = n_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = MaskedScaledDotProductAttention()

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # (batch, head, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        if mask is not None:
            # mask shape: (seq_len, seq_len) → broadcast용 확장
            mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, seq_len, seq_len)

        out, attn = self.attention(Q, K, V, mask)  # (batch, head, seq_len, d_k)

        # (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)


