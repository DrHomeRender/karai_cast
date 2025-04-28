import torch
import torch.nn as nn

import tensorlization
from util import *
import math

from util.Encoder import TransformerEncoderLayer
from util.PositionalEncoding import PositionalEncoding


# (단일 입력 시퀀스) → Encoder → Flatten → Regression

# 가장 기본이 되는 계산단위

# 여러개의 ScaledDotProductAttention 을 병렬로 수행하고 concat+projection 하는구조 머리를 여러개 둬서 다양한 관점을 보게하는 구조


class TransformerModel(nn.Module):
    """전체 Transformer 시계열 예측 모델"""
    def __init__(self, input_dim, output_dim, d_model=64, n_head=4, d_ff=128, num_layers=2, dropout=0.1, seq_len=10):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model * seq_len, output_dim)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding_layer(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)  # Flatten
        return self.output_layer(x)


# 테스트용 샘플 입력
if __name__ == "__main__":
    batch_size = 16
    seq_len = 10
    input_dim = 1
    output_dim = 1
    data_path = "data20000.csv"

    input_tensor=tensorlization.tensor_from_csv(data_path, seq_len=10)
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    output = model(input_tensor)

    print("Transformer 출력 shape:", output.shape)
