import numpy as np
import torch
import torch.nn as nn
def generateSquareSubsequentMask(seq_len):
    """
    미래 시점을 보지 못하게 가리는 look-ahead mask
    shape: (seq_len, seq_len)
    """
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return ~mask  # False → 가려짐 (0), True → 통과 (1)

# def autoregressive_inference(model, src, pred_len=5):
#     """
#     Auto-Regressive 방식으로 다음 pred_len개 시점 예측
#     - src: (1, seq_len, 1)
#     - 출력: (pred_len, 1)
#     """
#     model.eval()
#     device = next(model.parameters()).device
#     src = src.to(device)
#
#     # tgt 시작은 0으로 시작
#     tgt_seq = torch.zeros((1, 1, 1), dtype=torch.float32).to(device)
#
#     preds = []
#
#     for _ in range(pred_len):
#         # 마스크 생성
#         tgt_mask = generateSquareSubsequentMask(tgt_seq.size(1)).to(device)
#
#         # 모델에 입력
#         with torch.no_grad():
#             out = model(src, tgt_seq, tgt_mask)  # (1, 1)
#             next_val = out[:, -1:]  # 마지막 예측값만 사용 (마지막 시점)
#
#         preds.append(next_val.cpu().numpy())
#
#         # 예측값을 다시 입력 시퀀스에 누적
#         next_val = next_val.unsqueeze(-1)  # (1, 1, 1)
#         tgt_seq = torch.cat([tgt_seq, next_val], dim=1)
#
#     return np.concatenate(preds, axis=1).flatten()
def autoregressive_inference(model, src, pred_len):
    model.eval()
    predicted = []
    tgt_seq = torch.zeros_like(src)

    with torch.no_grad():
        for _ in range(pred_len):
            tgt_mask = generateSquareSubsequentMask(tgt_seq.size(1)).to(src.device)
            out = model(src, tgt_seq, tgt_mask)

            # ✅ 추론 시 output shape이 (1, 64)면 임시 Linear 적용
            if out.shape[-1] == 64 and len(out.shape) == 2:
                out = nn.Linear(64, 1).to(out.device)(out)

            predicted.append(out.squeeze().cpu().numpy())

            next_input = out.unsqueeze(1)  # (B, 1, 1)
            tgt_seq = torch.cat([tgt_seq, next_input], dim=1)

    return np.array(predicted)
