import torch
import torch.nn as nn
import os
import sys
import math
from util.tensorlization import tensor_from_csv
import torch.optim as optim

from util.Encoder import TransformerEncoderLayer
from util.PositionalEncoding import PositionalEncoding

from sklearn.model_selection import train_test_split


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# (단일 입력 시퀀스) → Encoder → Flatten → Regression

# 가장 기본이 되는 계산단위

# 여러개의 ScaledDotProductAttention 을 병렬로 수행하고 concat+projection 하는구조 머리를 여러개 둬서 다양한 관점을 보게하는 구조


class TransformerModel(nn.Module):
    """전체 Transformer 시계열 예측 모델"""
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
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        x = x.view(batch_size * seq_len, input_dim)
        x = self.embedding_layer(x)
        x = x.view(batch_size, seq_len, -1)

        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)  # (batch, d_model * seq_len)
        return self.output_layer(x)  # (batch, output_dim)


if __name__ == "__main__":
    # 데이터 로딩
    # input_tensor = tensor_from_csv("data20000.csv", seq_len=10)
    target_columns = ["expected_growth", "expected_ph_duration", "target_ph"]
    input_tensor, target = tensor_from_csv("data20000.csv", seq_len=10, target_cols=target_columns)
    output_dim = len(target_columns)

    input_dim = input_tensor.shape[2]
    output_dim = 3
    seq_len = 10

    # 라벨 임의 생성 (예: expected_growth를 라벨로 뽑는 게 정확하나, 지금은 더미용)
    target = torch.randn(input_tensor.shape[0], output_dim)  # 더미 라벨

    # 훈련/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(input_tensor, target, test_size=0.2, random_state=42)

    # 모델 정의
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 훈련 루프
    n_epochs = 5
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    # 모델 저장
    torch.save(model.state_dict(), "model.pth")
    print(" 모델 저장 완료: model.pth")
