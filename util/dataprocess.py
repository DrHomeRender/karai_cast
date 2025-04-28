import pandas as pd
import torch

def extract_features(df):
    numbers = df[['num1','num2','num3','num4','num5','num6']].values.astype(int)

    print("숫자", numbers,type(numbers))
    one_digit = sum(1 <= n <= 9  for n in numbers)
    ten = sum(10 <= n <= 19 for n in numbers)
    twenty = sum(20 <= n <=29 for n in numbers)
    thirty = sum(30 <= n <= 39 for n in numbers)
    forty = sum(40 <= n <= 49 for n in numbers)

    has_consecutive = int(any(b-1 == 1 for a,b in zip(sorted(numbers),sorted(numbers)[1:])))
    bonus = df['bonus']
    return [one_digit, ten, twenty,thirty, forty,has_consecutive,bonus]

def tensor_from_csv(csv_path, seq_len=10):
    df = pd.read_csv(csv_path)
    Object=df.columns[0]
    df = df.sort_values(Object)

    features = df.apply(extract_features, axis=1).tolist()
    features = torch.tensor(features, dtype=torch.float32)

    data = []
    for i in range(len(features) - seq_len):
        window = features[i:i + seq_len]
        data.append(window)

    return torch.stack(data)  # [batch, seq_len, input_dim]


csv_path = "data_l.csv"
df = pd.read_csv(csv_path)
df = df.sort_values("time")

features = df.apply(extract_features, axis=1).tolist()
print(features)