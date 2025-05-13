import torch
import pandas as pd
from util.tensorlization import tensor_from_csv
from train import TransformerModel
import matplotlib.pyplot as plt
import json
import argparse

def device_rule(target_temp_air=27,target_temp_water=57,target_humidity=21):
    pred_temp_air = df_pred["predicted_temp_air"]
    pred_temp_water = df_pred["predicted_temp_water"]
    pred_humidity = df_pred["predicted_humidity"]

    # 제어기 결정 (기준 미만 → 온도 증가 장치 ON / 기준 초과 → OFF)
    df_pred["device_temp_air"] = (pred_temp_air < target_temp_air).astype(int)
    df_pred["device_temp_water"] = (pred_temp_water < target_temp_water).astype(int)
    df_pred["device_humidity"] = (pred_humidity < target_humidity).astype(int)

    # 확인
    print(" 예측 기반 제어기 상태 (상위 10개):")
    print(df_pred[["time", "predicted_temp_air", "device_temp_air",
                   "predicted_temp_water", "device_temp_water",
                   "predicted_humidity", "device_humidity"]].head(10))

    # 저장
    df_pred.to_csv("predictions_with_control.csv", index=False)
    print(" 제어기 매핑 결과 저장 완료: predictions_with_control.csv")


if __name__ == '__main__':
    # 자동 로드
    with open("model_meta.json", "r") as f:
        meta_info = json.load(f)

    TARGET_COLS = meta_info["target_columns"]
    output_dim = meta_info["output_dim"]
    seq_len = meta_info["seq_len"]
    print(f"[INFO] 메타 정보 불러오기 완료. 타겟 컬럼: {TARGET_COLS}")

    # ------------------- 설정 -------------------
    parser = argparse.ArgumentParser(description="Transformer 시계열 예측 학습")
    parser.add_argument("-i", "--data_path", type=str, default="data20000.csv", help="학습 데이터 CSV 경로")
    parser.add_argument("-m", "--model", type=str, default="model.pth", help="모델 저장 경로")
    args = parser.parse_args()

    data_path = args.data_path
    model_path = args.model
    df = pd.read_csv(data_path)
    print(df.columns)

    # 전체 컬럼 보기 설정
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    # ------------------- 원본 데이터 로딩 (시간 매핑용) -------------------
    df_raw = pd.read_csv(data_path)
    if "time" not in df_raw.columns:
        raise ValueError(" 'time' 컬럼이 데이터에 없습니다.")
    time_list = df_raw["time"].iloc[seq_len:].reset_index(drop=True)

    # ------------------- 입력 텐서 생성 -------------------
    input_tensor, _ = tensor_from_csv(data_path, seq_len=seq_len, target_cols=TARGET_COLS)
    input_dim = input_tensor.shape[2]

    # ------------------- 모델 불러오기 -------------------
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    model.load_state_dict(torch.load(model_path))
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
    true_values = df_true[TARGET_COLS].iloc[seq_len:].reset_index(drop=True).values[:len(df_pred)]

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

    if not any("device" in col for col in TARGET_COLS):
        print("[INFO] 센서 예측 모델입니다. device_rule() 자동 실행합니다.")
        device_rule()
    else:
        print("[INFO] 장치 예측 모델입니다. device_rule() 호출 생략.")
    # ------------------- 추가 평가 (val 데이터 기준) -------------------
    use_val_data = input("x_val.csv / y_val.csv로 평가하시겠습니까? (y/n) > ").strip().lower()

    if use_val_data == "y":
        x_val = pd.read_csv("x_val.csv").values
        y_val = pd.read_csv("y_val.csv").values

        # 자동 계산
        input_dim_val = x_val.shape[1] // seq_len
        x_val_tensor = torch.tensor(x_val, dtype=torch.float32).view(-1, seq_len, input_dim_val)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        with torch.no_grad():
            pred_val = model(x_val_tensor).numpy()
            true_val = y_val_tensor.numpy()

        print(" x_val.csv 기반 예측 완료")

        for i, col in enumerate(TARGET_COLS):
            plt.figure(figsize=(10, 3))
            plt.plot(true_values[:, i], label="real (0/1)", linewidth=2)
            plt.plot(pred_values[:, i], label="pred (float)", linestyle='--')

            # Sigmoid 적용해서 예측을 이진화한 경우
            pred_binary = (pred_values[:, i] > 0.5).astype(int)
            plt.plot(pred_binary, label="pred (binary)", linestyle=':')

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

            plt.title(f"compare - {col} | Max Err: {max_error:.3f}, Min Err: {min_error:.3f}")
            plt.xlabel("time")
            plt.ylabel("value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"compare_val_{col}.png")
            plt.show()

