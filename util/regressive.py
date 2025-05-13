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

def autoregressive_inference(model, src, pred_len):
    model.eval()
    predicted = []
    tgt_seq = torch.zeros_like(src)

    with torch.no_grad():
        for _ in range(pred_len):
            tgt_mask = generateSquareSubsequentMask(tgt_seq.size(1)).to(src.device)
            out = model(src, tgt_seq, tgt_mask)

            #  추론 시 output shape이 (1, 64)면 임시 Linear 적용
            if out.shape[-1] == 64 and len(out.shape) == 2:
                out = nn.Linear(64, 1).to(out.device)(out)

            predicted.append(out.squeeze().cpu().numpy())

            next_input = out.unsqueeze(1)  # (B, 1, 1)
            tgt_seq = torch.cat([tgt_seq, next_input], dim=1)

    return np.array(predicted)
