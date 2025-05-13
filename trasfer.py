import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from util.tensorlization import tensor_from_csv
from train import TransformerModel
from sklearn.model_selection import train_test_split
import argparse

from util.finetune import FineTuneModel
# ---------- 메인 실행 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", type=str, default="data20000.csv", help="Fine-tune 데이터 CSV 경로")
    parser.add_argument("-p", "--pretrained_model", type=str, default="model.pth", help="사전 학습된 모델 경로")
    parser.add_argument("-s", "--seq_len", type=int, default=10, help="시퀀스 길이")
    parser.add_argument("-e", "--epoch", type=int, default=2000, help="Fine-tune Epoch 수")
    parser.add_argument("-o", "--output_model", type=str, default="fine_tuned_model.pth", help="Fine-tuned 모델 저장 경로")
    args = parser.parse_args()

    # 데이터 로딩
    df = pd.read_csv(args.data_path)
    print(f"[INFO] 데이터 컬럼: {list(df.columns[1:])}")

    print("[입력] Fine-tune할 새로운 타겟 컬럼을 ','로 구분해서 입력하세요:")
    target_input = input(">> ").strip()
    target_columns = [col.strip() for col in target_input.split(",")]

    # 텐서 생성
    input_tensor, target = tensor_from_csv(args.data_path, seq_len=args.seq_len, target_cols=target_columns)
    input_dim = input_tensor.shape[2]
    output_dim = len(target_columns)

    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(input_tensor, target, test_size=0.2, random_state=42)

    # 사전 학습된 모델 로드
    pretrained_model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=args.seq_len)
    pretrained_model.load_state_dict(torch.load(args.pretrained_model))
    print(f"[INFO] 사전 학습된 모델 불러오기 완료: {args.pretrained_model}")

    # Fine-tune 모델 구성
    fine_tune_model = FineTuneModel(pretrained_model, new_output_dim=output_dim)

    # Encoder Freeze 여부 선택
    freeze_encoder = input("[입력] Encoder를 고정하시겠습니까? (y/n) > ").strip().lower()
    if freeze_encoder == "y":
        for param in fine_tune_model.embedding_layer.parameters():
            param.requires_grad = False
        for layer in fine_tune_model.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        print("[INFO] Encoder 고정 (Freeze) 완료")
    else:
        print("[INFO] Encoder까지 Fine-tune 진행")

    # 훈련
    criterion = nn.MSELoss()
    optimizer = optim.Adam(fine_tune_model.parameters(), lr=0.001)

    for epoch in range(args.epoch):
        fine_tune_model.train()
        optimizer.zero_grad()
        output = fine_tune_model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        fine_tune_model.eval()
        with torch.no_grad():
            val_output = fine_tune_model(X_val)
            val_loss = criterion(val_output, y_val)

        if epoch % 100 == 0 or epoch == args.epoch - 1:
            print(f"Epoch {epoch+1}/{args.epoch} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    # 저장
    torch.save(fine_tune_model.state_dict(), args.output_model)
    print(f"[INFO] Fine-tuned 모델 저장 완료: {args.output_model}")
