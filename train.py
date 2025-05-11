import pandas as pd
import torch
import torch.nn as nn
import os
import numpy as np
import sys
import math
from util.tensorlization import tensor_from_csv
import torch.optim as optim
import argparse
from util.Encoder import TransformerEncoderLayer
from util.PositionalEncoding import PositionalEncoding
import matplotlib.pyplot as plt
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
        # x = x.view(batch_size * seq_len, input_dim)
        x = x.reshape(batch_size * seq_len, input_dim)

        x = self.embedding_layer(x)
        x = x.view(batch_size, seq_len, -1)

        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)  # (batch, d_model * seq_len)
        return self.output_layer(x)  # (batch, output_dim)


if __name__ == "__main__":
    # 입력부분

    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--data_path", type= str ,default ="data20000.csv")
    args = parser.parse_args()
    print("학습 데이터 경로",args.data_path)
    data_path = args.data_path
    df = pd.read_csv(data_path)
    print("데이터 통계는 다음과 같습니다 :" ,df.describe())
    print("학습할 데이터의 항목은 다음과 같습니다.:",list(df.columns[1:]))
    target_category = int(input("예측 할 항목은 몇개 입니까?"))
    # TODO : api 화를 위한 forecatst config 작성

    target_columns = df.columns[1:target_category+1]

    input_tensor, target = tensor_from_csv("data20000.csv", seq_len=10, target_cols=target_columns)
    output_dim = len(target_columns)


    input_dim = input_tensor.shape[2]
    output_dim = target_category
    seq_len = 10


    # 훈련/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(input_tensor, target, test_size=0.2, random_state=42)

    # numpy로 변환
    X_val_np = X_val.numpy().reshape(X_val.shape[0], -1)  # (batch, seq_len * input_dim)
    y_val_np = y_val.numpy()

    # 저장
    df_xval = pd.DataFrame(X_val_np)
    df_yval = pd.DataFrame(y_val_np, columns=[f"target_{i}" for i in range(y_val_np.shape[1])])

    df_xval.to_csv("x_val.csv", index=False)
    df_yval.to_csv("y_val.csv", index=False)
    print(" 검증용 데이터 저장 완료: x_val.csv, y_val.csv")

    # 모델 정의
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 훈련 루프
    n_epochs = int(input("얼마나 학습할건가요?"))
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
    print("예측 항목갯수는 :",target_category,"이고 test할때 기입하십쇼")
