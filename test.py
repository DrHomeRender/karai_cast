import torch
import pandas as pd
from util.tensorlization import tensor_from_csv
from train import TransformerModel
import matplotlib.pyplot as plt
import json
import argparse
import numpy as np
# 마이띵스 자체룰이 있으면 그걸 사용하시면 됩니다. 이 함수는 센서값만 있을때 0 ,1로 매핑 할 수있는 예시 입니다.
# 장치값이 존재 한다면 필요 없습니다.
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

    target_cols = meta_info["target_columns"]
    input_cols = meta_info["input_columns"]
    output_dim = meta_info["output_dim"]
    seq_len = meta_info["seq_len"]
    print(f"[INFO] 메타 정보 불러오기 완료. 타겟 컬럼: {target_cols}")

    # ------------------- 설정 -------------------
    parser = argparse.ArgumentParser(description="Transformer 시계열 예측 학습")
    parser.add_argument("-i", "--data_path", type=str, default="data20000.csv", help="학습 데이터 CSV 경로")
    parser.add_argument("-m", "--model", type=str, default="model1.pth", help="모델 저장 경로")
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
    input_tensor, _ = tensor_from_csv(data_path, seq_len=seq_len, input_cols=input_cols,target_cols=target_cols)
    input_dim = input_tensor.shape[2]

    # ------------------- 모델 불러오기 -------------------
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # ------------------- 예측 수행 -------------------
    with torch.no_grad():
        predictions = model(input_tensor).numpy()  # shape: (batch, output_dim)
    df_true = pd.read_csv(data_path)
    true_values = df_true[target_cols].iloc[seq_len:].reset_index(drop=True).values[:len(predictions)]

    #  device 예측시 → real 강제 0/1 매핑 및 저장
    if meta_info.get("task_type") == "device":
        df_real = pd.DataFrame(true_values, columns=target_cols)
        df_real.insert(0, "time", time_list)

        # 혹시 float로 들어가 있는 경우 → 0.5 기준 이진화
        for col in target_cols:
            df_real[col] = (df_real[col] > 0.5).astype(int)

        df_real.to_csv("real_with_control.csv", index=False)
        print("[INFO] 장치 예측 → 실제값 ON/OFF 기준으로 재매핑 저장 완료: real_with_control.csv")

    # ------------------- 결과 매핑 및 출력 -------------------
    col_names = [f"predicted_{col}" for col in target_cols]
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
    true_values = df_true[target_cols].iloc[seq_len:].reset_index(drop=True).values[:len(df_pred)]

    # 예측값
    pred_values = predictions[:len(true_values)]


    # ------------------- 시각화 -------------------
    for i, col in enumerate(target_cols):
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

    if not any("device" in col for col in target_cols):
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

        # ------------------- 시각화 -------------------
        for i, col in enumerate(target_cols):
            plt.figure(figsize=(10, 3))

            if meta_info.get("task_type") == "device":
                #  device 예측 → sigmoid 적용 + step plot
                sigmoid_pred = 1 / (1 + np.exp(-pred_values[:, i]))
                binary_pred = (sigmoid_pred > 0.5).astype(int)

                plt.step(range(len(binary_pred)), binary_pred, where='post', label="pred (binary)", linestyle='--',
                         color='orange')
                plt.step(range(len(true_values[:, i])), true_values[:, i], where='post', label="real (0/1)",
                         linestyle='-', color='blue')

                plt.title(f"🔌 Device On/Off Compare - {col}")
                plt.ylim(-0.2, 1.2)
                plt.yticks([0, 1], ['OFF', 'ON'])

            else:
                # sensor 예측 → 기존 방식
                plt.plot(true_values[:, i], label="real", linewidth=2)
                plt.plot(pred_values[:, i], label="pred", linestyle='--')
                plt.title(f" sensor_pred_compare - {col}")

            # 오차 표시 (공통)
            errors = abs(true_values[:, i] - pred_values[:, i])
            max_error = errors.max()
            min_error = errors.min()
            max_idx = errors.argmax()

            plt.scatter(max_idx, true_values[max_idx, i], color='red', label='max error (real)', zorder=5)
            plt.scatter(max_idx, pred_values[max_idx, i], color='orange', label='max error (pred)', zorder=5)

            plt.annotate(f"{true_values[max_idx, i]:.2f}", (max_idx, true_values[max_idx, i]),
                         textcoords="offset points", xytext=(0, 10), ha='center', color='red')
            plt.annotate(f"{pred_values[max_idx, i]:.2f}", (max_idx, pred_values[max_idx, i]),
                         textcoords="offset points", xytext=(0, -15), ha='center', color='orange')

            plt.xlabel("time")
            plt.ylabel("value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"compare_{col}.png")
            plt.show()

        p
