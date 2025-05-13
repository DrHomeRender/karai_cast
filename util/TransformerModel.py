


import torch.nn as nn
from .Encoder import TransformerEncoderLayer
from .PositionalEncoding import PositionalEncoding


class TransformerModel(nn.Module):
    """Transformer 기반 시계열 예측 모델"""
    def __init__(self, input_dim, output_dim, d_model=64, n_head=4, d_ff=128, num_layers=2, dropout=0.1, seq_len=10):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, d_model)
        print("[DEBUG] embedding weight shape:", self.embedding_layer.weight.shape)

        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model * seq_len, output_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.reshape(batch_size * seq_len, input_dim)
        x = self.embedding_layer(x)
        x = x.view(batch_size, seq_len, -1)

        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)
        return self.output_layer(x)

