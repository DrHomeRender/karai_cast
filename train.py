import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sklearn.model_selection import train_test_split
from util.tensorlization import tensor_from_csv
from util.TransformerModel import TransformerModel

# ----------------------- 메인 실행 -----------------------
if __name__ == "__main__":
    # ---------- 입력 파라미터 ----------
    parser = argparse.ArgumentParser(description="Transformer 시계열 예측 학습")
    parser.add_argument("-p", "--data_path", type = str, default = "data20000.csv", help = "학습 데이터 CSV 경로")
    parser.add_argument("-s","--seq_len", type = int, default = 10 , help = "시퀀스 길이")
    parser.add_argument("-o","--model", type = str, default = "model.pth", help ="모델 저장 경로")
    args = parser.parse_args()

    # ---------- 데이터 로드 및 정보 ----------
    print(f"\n[INFO] 학습 데이터 경로: {args.data_path}")
    df = pd.read_csv(args.data_path)
    print("[INFO] 데이터 통계:\n", df.describe())
    print("[INFO] 학습 대상 컬럼:", list(df.columns[1:]))

    # ---------- 예측 타겟 설정 ----------
    # TODO: 좀더 좋은 방법 구상 순차가 아닌 선택으로
    target_category = int(input("\n[입력] 예측할 항목 수를 입력하세요: "))
    target_columns = df.columns[1:target_category+1]
    print(f"[INFO] 예측할 타겟 컬럼: {list(target_columns)}")

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

    # ---------- 학습 ----------
    n_epochs = int(input("\n[입력] 학습할 에폭 수를 입력하세요: "))
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
