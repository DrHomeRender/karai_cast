#  test_half_forecasting.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PositionalEncoding import *
from createSlidingWindowData import *
from Encoder import *
from Decoder import *
from train import TransformerModel
from regressive import autoregressive_inference

#  하이퍼파라미터 설정
input_dim = 1
output_dim = 1

#  전체 사인파 불러오기
df = pd.read_csv("../data/sin_data.csv")
values = df["value"].values  # (1000,)
total_len = len(values)
half_len = total_len // 2

#  앞 절반만 입력
src_input = values[:half_len]  # (500,)
src = torch.tensor(src_input, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, 500, 1)

#  예측 길이: 뒤 절반
pred_len = total_len - half_len  # = 500

#  모델 생성 (입력 길이에 맞춰 seq_len 설정)
model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=half_len)
model.load_state_dict(torch.load("transformer_sine_500.pt"))
model.eval()

#  Auto-Regressive 예측
predicted_seq = autoregressive_inference(model, src, pred_len)

#  시각화
plt.figure(figsize=(12, 5))
plt.plot(np.arange(half_len), src.squeeze().numpy(), label="Input Sequence", color="blue")
plt.plot(np.arange(half_len, total_len), predicted_seq, label="Predicted Future", color="red", linestyle="--")
plt.title("Auto-Regressive Forecasting (Half Sine Input → Full Sine Prediction)")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
