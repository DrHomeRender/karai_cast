
from PositionwiseFeedForward import *
from maskSelfAttention import *
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MaskedMultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        # 멀티헤드 어텐션 누락
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Masked Multi-Head Self-Attention
        attn_out = self.self_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Positionwise FeedForward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, num_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)