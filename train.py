import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
from util.tensorlization import tensor_from_csv
from util.TransformerModel import TransformerModel
import json
import os
# ----------------------- 메인 실행 -----------------------
if __name__ == "__main__":
    # ---------- 입력 파라미터 ----------
    parser = argparse.ArgumentParser(description="Transformer 시계열 예측 학습")
    parser.add_argument("-i", "--data_path", type = str, default = "data20000.csv", help = "학습 데이터 CSV 경로")
    parser.add_argument("-s","--seq_len", type = int, default = 10 , help = "시퀀스 길이") # 데이터증가 할경우 증가
    parser.add_argument("-o","--model", type = str, default = "model.pth", help ="모델 저장 경로")
    parser.add_argument("-e","--epoch", type = int, default = 5000, help ="epoch 수")
    args = parser.parse_args()

    # ---------- 데이터 로드 및 정보 ----------
    print(f"\n[INFO] 학습 데이터 경로: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print("[INFO] 데이터 통계:\n", df.describe())
    print("[INFO] 학습 대상 컬럼:", list(df.columns[1:]))

    # ---------- 예측 타겟 설정 ---------- 카테고리
    # ---------- 선택 방식 ----------
    print("\n[옵션] 예측 컬럼 선택 방식을 골라주세요:")
    print("1. 직접 컬럼명 입력")
    print("2. 카테고리 선택 후 컬럼명 선택")
    mode_select = input("\n[입력] 선택하세요 (1/2): ").strip()

    if mode_select == "1":
        #  직접 입력 모드
        print("[입력] 예측할 컬럼명을 ','로 구분하여 입력하세요 (예: temp_air,temp_water):")
        target_input = input(">> ").strip()
        target_columns = [col.strip() for col in target_input.split(",")]

        missing_cols = [col for col in target_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"[ERROR] 존재하지 않는 컬럼명 입력: {missing_cols}")

    else:
        #  자동 분류 + 카테고리 선택 모드
        sensor_cols = []
        device_cols = []
        expected_cols = []
        target_cols = []

        for col in df.columns[1:]:
            if "device" in col:
                device_cols.append(col)
            elif "expected" in col:
                expected_cols.append(col)
            elif "target" in col:
                target_cols.append(col)
            else:
                sensor_cols.append(col)

        print("\n[INFO] 자동 분류 결과")
        print("1. 센서(sensor):", sensor_cols)
        print("2. 제어(device):", device_cols)
        print("3. 예상(expected):", expected_cols)
        print("4. 기타(target):", target_cols)

        category_map = {
            "1": sensor_cols,
            "2": device_cols,
            "3": expected_cols,
            "4": target_cols
        }

        selected_category = input("\n[입력] 카테고리 번호를 선택하세요 (1/2/3/4): ").strip()
        if selected_category not in category_map:
            raise ValueError("[ERROR] 올바른 번호를 선택하세요 (1/2/3/4)")

        print(f"\n[INFO] 선택 가능한 컬럼: {category_map[selected_category]}")
        print("[입력] 예측할 컬럼명을 ','로 구분하여 입력하세요:")
        target_input = input(">> ").strip()
        target_columns = [col.strip() for col in target_input.split(",")]

        missing_cols = [col for col in target_columns if col not in category_map[selected_category]]
        if missing_cols:
            raise ValueError(f"[ERROR] 잘못된 컬럼 입력: {missing_cols}")

    print(f"[INFO] 최종 예측할 타겟 컬럼: {target_columns}")

    # ---------- 텐서 생성 ----------
    input_tensor, target = tensor_from_csv(args.data_path, seq_len=10, target_cols=target_columns)
    input_dim = input_tensor.shape[2]
    output_dim = len(target_columns)
    seq_len = args.seq_len

    # ---------- 훈련/검증 데이터 분할 ----------
    X_train, X_val, y_train, y_val = train_test_split(input_tensor, target, test_size=0.2, random_state=42)
    print(f"[INFO] 훈련 데이터: {X_train.shape}, 검증 데이터: {X_val.shape}")

    # ---------- 검증 데이터 CSV 저장 ----------
    df_xval = pd.DataFrame(X_val.numpy().reshape(X_val.shape[0], -1))
    df_yval = pd.DataFrame(y_val.numpy(), columns=[f"target_{i}" for i in range(y_val.shape[1])])
    df_xval.to_csv("x_val.csv", index=False)
    df_yval.to_csv("y_val.csv", index=False)
    print("[INFO] 검증용 데이터 저장 완료: x_val.csv, y_val.csv")

    # ---------- 모델 생성 ----------
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------- 이어서 학습 여부 확인 ----------
    resume_training = False
    if os.path.exists(args.model) and os.path.exists("model_meta.json"):
        resume_input = input(f"\n[INFO] 이전 학습 모델과 메타 정보가 존재합니다. 이어서 학습하시겠습니까? (y/n) > ").strip().lower()
        if resume_input == "y":
            resume_training = True

    # ---------- 데이터 로드 및 정보 ----------
    df = pd.read_csv(args.data_path)

    if resume_training:
        # 메타 정보 로드
        with open("model_meta.json", "r") as f:
            meta_info = json.load(f)

        target_columns = meta_info["target_columns"]
        input_tensor, target = tensor_from_csv(args.data_path, seq_len=meta_info["seq_len"], target_cols=target_columns)
        input_dim = meta_info["input_dim"]
        output_dim = meta_info["output_dim"]
        seq_len = meta_info["seq_len"]

        print(f"[INFO] 이어서 학습 타겟 컬럼: {target_columns}")
    else:
        # 처음부터 타겟 설정
        # (위에서 개선한 센서/디바이스 자동 구분 선택 코드와 결합 가능)
        target_category = int(input("\n[입력] 예측할 항목 수를 입력하세요: "))
        target_columns = df.columns[1:target_category + 1]

        input_tensor, target = tensor_from_csv(args.data_path, seq_len=args.seq_len, target_cols=target_columns)
        input_dim = input_tensor.shape[2]
        output_dim = len(target_columns)
        seq_len = args.seq_len

    # ---------- 모델 준비 ----------
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)

    if resume_training:
        model.load_state_dict(torch.load(args.model))
        print(f"[INFO] 모델 파라미터 불러오기 완료: {args.model}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ---------- 학습 ----------
    n_epochs = args.epoch
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        # 검증
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # ---------- 모델 저장 ----------
    torch.save(model.state_dict(), args.model)
    print(f"\n모델 저장 완료: {args.model}")
    print(f" 예측 항목 수: {output_dim} | 테스트 시 입력하세요.")

    # ---------- 메타 데이터 저장 ----------
    # 마지막 저장 전에 추가
    meta_info = {
        "target_columns": target_columns,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "seq_len": seq_len
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta_info, f, indent=4)

    print("메타 정보 저장 완료: model_meta.json")
