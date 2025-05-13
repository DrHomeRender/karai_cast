import torch

import pandas as pd
import matplotlib.pyplot as plt

def control_forecast(row):

    return [
        row["temp_air"],
        row["temp_water"],
        row["humidity"],
        row["ph"],
        row["light"],
        row["salinity"],
        row["do"],
        row["co2_air"],
        row["set_temp_air"],
        row["set_ph"],
        row["set_oxygen"],
        row["target_growth"],
        row["target_ph"],
        row["tank_vol"],
        row["building_vol"],
        row["growth_len"],
        row["device_temp_air"],
        row["device_temp_water"],
        row["device_light"],
        row["device_ph_injection"],
        row["device_nitrogen"],
        row["device_oxygen"]
    ]


class data_analysis():
    def __init__(self,df):
        self.df=df

    def data_len(self):
        data_len=self.df.shape[0]
        print("데이터의 총량:", data_len)

    def data_info(self):
        data_info=self.df.info()
        print("데이터 정보",data_info)
        print("데이터 타입:")
        print(self.df.dtypes)

        print(" 결측치 수:")
        print(self.df.isnull().sum())

        print("기본 통계 요약:")
        print(self.df.describe())


    def data_visualization(self, x_col: str = None, y_col: str = None,index=1):
        """
        데이터 분포를 시각화합니다.
        기본적으로 첫 번째 열을 x축, 두 번째 열을 y축으로 사용합니다.
        """
        if x_col is None: # 시간 x축
            x_col = self.df.columns[0]
        if y_col is None: # 보고 싶은 데이터
            y_col = self.df.columns[index]

        if x_col not in self.df.columns or y_col not in self.df.columns:
            print(f" 지정한 컬럼이 존재하지 않습니다: {x_col}, {y_col}")
            return
        plt.figure(figsize=(10, 6))
        plt.plot(self.df[x_col], self.df[y_col], marker='o', linestyle='-', markersize=2)
        plt.title(f"{x_col} vs {y_col} data_graph")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.show()
class data_preprocess():
    def __init__(self,df):
        self.df=df

    def handle_missing(self, method='mean'):
        missing_count = self.df.isnull().sum().sum()

        if missing_count == 0:
            print(" 결측치 없음")
            return
        else:
            print(f" 결측치 탐지: {missing_count}개")

            # 결측치 처리

            if method == 'drop':
                self.df = self.df.dropna()
                print("결측치 제거 완료")
            elif method == "mean":
                self.df = self.df.fillna(self.df.mean(numeric_only=True))
                print("결측치를 평균값으로 대체")
            elif method == 'fill':
                self.df = self.df.fillna(method = 'ffill')
                print('결측치를 직전 값으로 채움')
            elif method == 'bfill':
                self.df = self.df.fillna(method = 'bfill')
                print("결측치를 그 다음 값으로 채움")
            else:
                print(f"지원하지 않는 방식!:{method}")

    def detect_outliers(self, col: str):

        print("지정열의.이상치를 탐지하고 범위를 출력합니다.")
        print("IQR 방식을 사용")

        if col not in self.df.columns:
            print(f"지정한 칼럼이 존재하지 않음: {col}")
        # 범위 지정
        q1 =self.df[col].quantile(0.25)
        q3 = self.df[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = self. df[(self.df[col] < lower_bound) & (self.df[col] > upper_bound)]
        print(f"이상치 감지 완료 -{col}")
        print(f"하한값 {lower_bound:.3f}, 상한값: {upper_bound:.3f}")
        print(f"이상치 개수 {outliers.shape[0]}개")

        if outliers.shape[0] > 0:
            print(outliers[[col]])

def tensor_from_csv_for_one(csv_path, seq_len=10, target_cols=["expected_growth"]):
    df = pd.read_csv(csv_path)
    df = df.sort_values(df.columns[0])  # 시간 기준 정렬

    # 특징 추출 (전처럼 control_forecast 등 사용)
    features = df.apply(control_forecast, axis=1).tolist()
    features = torch.tensor(features, dtype=torch.float32)

    # 타겟도 슬라이딩
    targets = df[target_cols].iloc[seq_len:].values
    targets = torch.tensor(targets, dtype=torch.float32)

    data = []
    for i in range(len(features) - seq_len):
        window = features[i:i + seq_len]
        data.append(window.unsqueeze(0))

    input_tensor = torch.cat(data, dim=0)  # (batch, seq_len, input_dim)
    return input_tensor, targets
def tensor_from_csv(csv_path, seq_len=10, input_cols=None, target_cols=None):
    import pandas as pd
    import torch

    df = pd.read_csv(csv_path)
    df = df.sort_values(df.columns[0])  # 시간 기준 정렬

    if input_cols is None or target_cols is None:
        raise ValueError("input_cols 와 target_cols 를 명시하세요.")

    # 입력 features
    features = df[input_cols].values
    features = torch.tensor(features, dtype=torch.float32)

    # 타겟 values
    targets = df[target_cols].iloc[seq_len:].values
    targets = torch.tensor(targets, dtype=torch.float32)

    data = []
    for i in range(len(features) - seq_len):
        window = features[i:i + seq_len]
        data.append(window.unsqueeze(0))

    input_tensor = torch.cat(data, dim=0)
    return input_tensor, targets

