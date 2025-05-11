import torch
import pandas as pd
from util.tensorlization import tensor_from_csv
from train import TransformerModel
import matplotlib.pyplot as plt
# ------------------- 설정 -------------------
SEQ_LEN = 10
DATA_PATH = "data20000.csv"
MODEL_PATH = "model.pth"
df = pd.read_csv(DATA_PATH)
target_category = int(input("예측 할 항목은 몇개 였나요?"))
TARGET_COLS = df.columns[1:target_category+1]

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

# ------------------- 실제값 로딩 -------------------
df_true = pd.read_csv("data20000.csv")
true_values = df_true[TARGET_COLS].iloc[SEQ_LEN:].reset_index(drop=True).values[:len(df_pred)]

# 예측값
pred_values = predictions[:len(true_values)]


# ------------------- 시각화 -------------------
for i, col in enumerate(TARGET_COLS):
    plt.figure(figsize=(10, 3))
    plt.plot(true_values[:, i], label="real", linewidth=2)
    plt.plot(pred_values[:, i], label="pred", linestyle='--')

    # 오차 계산
    errors = abs(true_values[:, i] - pred_values[:, i])
    max_error = errors.max()
    min_error = errors.min()
    max_idx = errors.argmax()

    # 마커 표시
    plt.scatter(max_idx, true_values[max_idx, i], color='red', label='max error (real)', zorder=5)
    plt.scatter(max_idx, pred_values[max_idx, i], color='orange', label='max error (pred)', zorder=5)

    # 수치 주석 표시
    plt.annotate(f"{true_values[max_idx, i]:.2f}", (max_idx, true_values[max_idx, i]),
                 textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    plt.annotate(f"{pred_values[max_idx, i]:.2f}", (max_idx, pred_values[max_idx, i]),
                 textcoords="offset points", xytext=(0, -15), ha='center', color='orange')

    # 타이틀 & 기타 설정
    plt.title(f" pred_compare - {col} | Max Err: {max_error:.3f}, Min Err: {min_error:.3f}")
    plt.xlabel("time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"compare_{col}.png")
    plt.show()

print("예측 결과 vs 실제값 비교 그래프 저장 완료 (compare_*.png)")

# ------------------- 추가 평가 (val 데이터 기준) -------------------
use_val_data = input("x_val.csv / y_val.csv로 평가하시겠습니까? (y/n) > ").strip().lower()

if use_val_data == "y":
    x_val = pd.read_csv("x_val.csv").values
    y_val = pd.read_csv("y_val.csv").values

    # 자동 계산
    input_dim_val = x_val.shape[1] // SEQ_LEN
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32).view(-1, SEQ_LEN, input_dim_val)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    with torch.no_grad():
        pred_val = model(x_val_tensor).numpy()
        true_val = y_val_tensor.numpy()

    print(" x_val.csv 기반 예측 완료")

    for i, col in enumerate(TARGET_COLS):
        plt.figure(figsize=(10, 3))
        plt.plot(true_val[:, i], label="real", linewidth=2)
        plt.plot(pred_val[:, i], label="pred", linestyle='--')

        # 오차 계산
        errors = abs(true_val[:, i] - pred_val[:, i])
        max_error = errors.max()
        min_error = errors.min()
        max_idx = errors.argmax()

        # 마커 표시
        plt.scatter(max_idx, true_val[max_idx, i], color='red', label='max error (real)', zorder=5)
        plt.scatter(max_idx, pred_val[max_idx, i], color='orange', label='max error (pred)', zorder=5)

        # 수치 주석 표시
        plt.annotate(f"{true_val[max_idx, i]:.2f}", (max_idx, true_val[max_idx, i]),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='red')
        plt.annotate(f"{pred_val[max_idx, i]:.2f}", (max_idx, pred_val[max_idx, i]),
                     textcoords="offset points", xytext=(0, -15), ha='center', color='orange')

        plt.title(f"🧪 compare - {col} | Max Err: {max_error:.3f}, Min Err: {min_error:.3f}")
        plt.xlabel("time")
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"compare_val_{col}.png")
        plt.show()

