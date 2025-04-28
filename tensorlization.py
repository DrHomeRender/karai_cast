import torch
import torch.nn as nn
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from util import PositionalEncoding
from pandas.io.xml import preprocess_data


# TODO : 모델 설계 할때 Transformer base LTSF-Linear 혹은 LSTM transfromer

class feature_extractor():
    def __init__(self):
        super(feature_extractor, self).__init__()

    def lotto_analysis(df): # 특징 축출 함수
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

    def control_forecasst(df):
        print(df.head)
        print(df.columns)
        print(df.info(5))


        #TODO : 설계
        print("제어 예측을 위한 전처리")
class data_process():
    def __init__(self,df):
        self.df=df
    # 1. 현재 데이터의 수를 출력한다.(o)
    def len_data(self):
        data_len=self.df.shape[0]
        print("data_len:", data_len)
    # 2. 데이터 타입 자동 변환 (o)
    def data_type(self):
        data_type=self.df.dtype
        print("data_type:", data_type)
    # 3. y값 분포 시각화 (o)
    def visualization_data(self):
        x_axis=self.df.iloc[:,0]
        y_axis=self.df.iloc[:,1]
        plt.plot(x_axis, y_axis)
        plt.show()

    # 4. 결측값 분석 자동 수정 (o)
    def data_missing(self,method):
        replace_data_nan = self.df.replace(r'^\s*$', np.nan, regex=True)
        if method == 'drop':
            self.df.dropna()
        elif method == 'mean':
            df_mean = self.df.fillna(self.df.mean(numeric_only=True))
        else:
            df_mean = self.df.fillna(self.df.mean())


    # 5. 이상치 분석 자동
    def data_singurality(self,method):
        replace_data_nan = self.df.replace(r'^\s*$', np.nan, regex=True)
        if method == 'drop':
            self.df.dropna()

    # 6 중복데이터 확인 제거
    def data_confirm_duplication(self,method):
        replace_data_nan = self.df.replace(r'^\s*$', np.nan, regex=True)
        if method == 'drop':
            self.df.dropna()
        elif method == 'mean':
            df_mean = self.df.fillna(self.df.mean(numeric_only=True))


    # 7 데이터 분포 및 불균형 확인 시각화
    def unbelance_data_visualization(self,method):
        x_axis=self.df.iloc[:,0]
        y_axis=self.df.iloc[:,1]
        plt.plot(x_axis, y_axis)
        plt.show()
    # 8 샘플링 여부 판단
    def check_data_sampling(self):
        check_data = self.df.iloc[:,0]
        numpy_conversion = np.array(self.df.iloc[:,1])
        print("numpy_conversion:", numpy_conversion)
        print("check_data:", check_data)
        check_sampling = True
        if check_sampling:
            print("sampling ok")
        else:
            print("sampling request")

    # 9 시계
    def analysis_time_series_data(self):
        # Check data DMS or IMS
        DMS = self.df.iloc[:,0]
        DMS_filter = DMS < 0.1
        IMS = self.df.iloc[:,1]
        IMS_filter = IMS > 0.1

        if DMS_filter:
            DMS_



        if self.df == True:
            print("time series data")
        else:
            print("quantization data")


# TODO: 키워드 구현

    '''
    10. 시계열 데이터에관한 분석 Are Transformers Effective for Time Series Forecasting? 논문 기반
    10_1 DMS IMS 분류
    10_2 time series decomposition 3가지 성분 분해  trend seasonal residual
    10_3 이동 평균 moving average 이건 뭔지 알아봐야함
    
    '''

def tensor_from_csv(csv_path, seq_len=10): #텐서화
    df = pd.read_csv(csv_path)
    time=df.columns[0]
    df = df.sort_values(time)
    preprocess_data = df.apply(data_preprocess(df))
    features = df.apply(feature_extractor.control_forecasst(df), axis=1).tolist()
    print("특징 축출",features)
    features = torch.tensor(features, dtype=torch.float32)
    print("tensor shape:", features.shape)

    data = []
    for i in range(len(features) - seq_len):
        window = features[i:i + seq_len]
        data.append(window)
    stack=torch.stack(data)
    print("stack shape:", stack.shape)# [batch, seq_len, input_dim]
    return stack



if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 7
    d_model = 32

    # 2-1. Linear Embedding 임베딩 임력 특징 을 차원으로 변환
    embedding_layer = nn.Linear(input_dim, d_model)
    # sample_input = torch.randn(batch_size, seq_len, input_dim)
    tensor = tensor_from_csv('data_l.csv', seq_len=10)
    embedded = embedding_layer(tensor)  # [batch, seq_len, d_model] 32개 차원 펼침
    print(d_model,"차원 embedded shape:", embedded.shape)
    # # 2-2. Positional Encoding # 임베딩된 벡터에 위치정보 추가
    pos_encoder = PositionalEncoding(d_model=d_model)
    encoded = pos_encoder(embedded)  # [batch, seq_len, d_model]

    print(" 차원을 포지셔닝 인코딩","batch:",encoded.shape[0],"seq_len",encoded.shape[1],"dimension"
          ,encoded.shape[2])  # 예상: torch.Size([4, 10, 32])