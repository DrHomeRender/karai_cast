import torch
import torch.nn as nn
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import PositionalEncoding
from pandas.io.xml import preprocess_data
import time


# TODO : 모델 설계 할때 Transformer base LTSF-Linear 혹은 LSTM transfromer

class feature_extractor:
    def __init__(self, df: pd.DataFrame):
        self.df = df

        self.time = df.columns[0]  # 시간 값 0열
        self.sensor = df.columns[1:12].tolist() # 환경 센서값 1열~ 11열
        self.target_setting = df.columns[12:14].tolist() # 목표 성장량 및 목표 ph농도
        self.fixed_env= df.columns[14:16].tolist() # 탱크나 빌딩 같은 구조물
        self.current_growth_col = df.columns[16] # 현재 성장길이
        self.device_control = df.columns[17:23].tolist() # 제어 장치 결정값
        self.expect_growth = df.columns[23]# 예측 성장량
        self.expect_harvest = df.columns[24]#  예측 수확시기

    # def lotto_analysis(df): # 특징 축출 함수
    #     numbers = df[['num1','num2','num3','num4','num5','num6']].values.astype(int)
    #
    #     print("숫자", numbers,type(numbers))
    #     one_digit = sum(1 <= n <= 9  for n in numbers)
    #     ten = sum(10 <= n <= 19 for n in numbers)
    #     twenty = sum(20 <= n <=29 for n in numbers)
    #     thirty = sum(30 <= n <= 39 for n in numbers)
    #     forty = sum(40 <= n <= 49 for n in numbers)
    #
    #     has_consecutive = int(any(b-1 == 1 for a,b in zip(sorted(numbers),sorted(numbers)[1:])))
    #     bonus = df['bonus']
    #     return [one_digit, ten, twenty,thirty, forty,has_consecutive,bonus]

def control_forecast(row):
    # print(row.head)
    # print(row.columns)
    # print(row.info(5))


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
'''
    # 데이터 중복 제거
    # def data_duplication(self):
    #     return 0

    # 2. 데이터 타입 자동 변환 (o)
    # def data_type(self):
    #     data_type=self.df.dtype
    #     print("data_type:", data_type)

#     # 7 데이터 분포 및 불균형 확인 시각화
#     def unbelance_data_visualization(self,method):
#         x_axis=self.df.iloc[:,0]
#         y_axis=self.df.iloc[:,1]
#         plt.plot(x_axis, y_axis)
#         plt.show()

#     # 8 샘플링 여부 판단
#     def check_data_sampling(self):
#         check_data = self.df.iloc[:,0]
#         numpy_conversion = np.array(self.df.iloc[:,1])
#         print("numpy_conversion:", numpy_conversion)
#         print("check_data:", check_data)
#         check_sampling = True
#         if check_sampling:
#             print("sampling ok")
#         else:
#             print("sampling request")
#
#     # 9 시계
#     def analysis_time_series_data(self):
#         # Check data DMS or IMS
#         DMS = self.df.iloc[:,0]
#         DMS_filter = DMS < 0.1
#         IMS = self.df.iloc[:,1]
#         IMS_filter = IMS > 0.1
#
#         if DMS_filter:
#             DMS_
#
#
#
#         if self.df == True:
#             print("time series data")
#         else:
#             print("quantization data")
#
#
# # TODO: 키워드 구현
#
#     
#     10. 시계열 데이터에관한 분석 Are Transformers Effective for Time Series Forecasting? 논문 기반
#     10_1 DMS IMS 분류
#     10_2 time series decomposition 3가지 성분 분해  trend seasonal residual
#     10_3 이동 평균 moving average 이건 뭔지 알아봐야함
#
#     
'''

def tensor_from_csv(csv_path, seq_len=10, target_cols=["expected_growth"]):
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

#
# if __name__ == "__main__":
#     batch_size = 4
#     seq_len = 10
#     input_dim = 7
#     d_model = 32
#     data_path= '../data20000.csv'
#     print("데이터를 불러옵니다. 데이터 경로",data_path)
#     df = pd.read_csv(data_path)  # 예시
#     print("데이터가 학습에 적합한지 분석합니다.")
#     time.sleep(2)
#
#     analysis = data_analysis(df)
#     print("데이터 정보를 불러옵니다.")
#     # time.sleep(2)
#     # analysis.len_data()
#     time.sleep(2)
#
#     analysis.data_info()
#     #1~
#     print("데이터를 시각화 합니다")
#     analysis.data_visualization(index=2)
#
#
#     preprocess=data_preprocess(df)
#     # 결측치 처리 (평균값으로 채우기)
#     print("결측치를 처리합니다.")
#     preprocess.handle_missing(method='mean')
#
#     #  특정 컬럼 이상치 탐지
#     print("이상치를 탐지중")
#     preprocess.detect_outliers(col='temp_air')
#
#     # processor.data_type()
#
#     # # 2-1. Linear Embedding 임베딩 임력 특징 을 차원으로 변환
#     # embedding_layer = nn.Linear(input_dim, d_model)
#     # # sample_input = torch.randn(batch_size, seq_len, input_dim)
#     tensor = tensor_from_csv('data_l.csv', seq_len=10)
#     # embedded = embedding_layer(tensor)  # [batch, seq_len, d_model] 32개 차원 펼침
#     # print(d_model,"차원 embedded shape:", embedded.shape)
#     # # # 2-2. Positional Encoding # 임베딩된 벡터에 위치정보 추가
#     # pos_encoder = PositionalEncoding(d_model=d_model)
#     # encoded = pos_encoder(embedded)  # [batch, seq_len, d_model]
#     #
#     # print(" 차원을 포지셔닝 인코딩","batch:",encoded.shape[0],"seq_len",encoded.shape[1],"dimension"
#     #       ,encoded.shape[2])  # 예상: torch.Size([4, 10, 32])