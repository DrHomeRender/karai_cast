import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from util.tensorlization import tensor_from_csv
from train import TransformerModel
from sklearn.model_selection import train_test_split
import argparse
import os

# ---------- Fine-tune용 새로운 모델 정의 ----------
class FineTuneModel(nn.Module):
    def __init__(self, pretrained_encoder, new_output_dim):
        super().__init__()
        self.embedding_layer = pretrained_encoder.embedding_layer
        self.pos_encoder = pretrained_encoder.pos_encoder
        self.encoder_layers = pretrained_encoder.encoder_layers
        self.seq_len = pretrained_encoder.seq_len

        # 새 task용 output layer
        self.new_output_layer = nn.Linear(self.seq_len * self.embedding_layer.out_features, new_output_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.reshape(batch_size * seq_len, input_dim)
        x = self.embedding_layer(x)
        x = x.view(batch_size, seq_len, -1)

        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)
        return self.new_output_layer(x)
