import torch
import pandas as pd
from util.tensorlization import tensor_from_csv
from train import TransformerModel

# ------------------- 설정 -------------------
SEQ_LEN = 10
DATA_PATH = "data20000.csv"
MODEL_PATH = "model.pth"
TARGET_COLS = ["expected_growth", "expected_ph_duration", "target_ph"]

# 전체 컬럼 보기 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ------------------- 원본 데이터 로딩 (시간 매핑용) -------------------
df_raw = pd.read_csv(DATA_PATH)
if "time" not in df_raw.columns:
    raise ValueError(" 'time' 컬럼이 데이터에 없습니다.")
time_list = df_raw["time"].iloc[SEQ_LEN:].reset_index(drop=True)

# ------------------- 입력 텐서 생성 -------------------
input_tensor, _ = tensor_from_csv(DATA_PATH, seq_len=SEQ_LEN, target_cols=TARGET_COLS)
input_dim = input_tensor.shape[2]
output_dim = len(TARGET_COLS)

# ------------------- 모델 불러오기 -------------------
model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=SEQ_LEN)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ------------------- 예측 수행 -------------------
with torch.no_grad():
    predictions = model(input_tensor).numpy()  # shape: (batch, output_dim)

# ------------------- 결과 매핑 및 출력 -------------------
col_names = [f"predicted_{col}" for col in TARGET_COLS]
df_pred = pd.DataFrame(predictions, columns=col_names)
df_pred.insert(0, "time", time_list)
df_pred = df_pred.round(3)

print(" 예측 결과 (상위 10개):")
print(df_pred.head(10))

# ------------------- CSV 저장 -------------------
df_pred.to_csv("predictions_multi.csv", index=False)
print(" 다중 예측 결과 저장 완료: predictions_multi.csv")
