# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:34:10 2024

@author: sapzzil
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, mode
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 종목 리스트
stock_list = ['MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSM', 'JPM', 'TSLA', 'WMT']
start_date = "2020-01-01"
end_date = "2023-12-31"

# 데이터 가져오기
def fetch_stock_data(stock_list, start, end):
    all_data = {}
    for stock in stock_list:
        all_data[stock] = yf.download(stock, start=start, end=end)
    return all_data

stock_data = fetch_stock_data(stock_list, start_date, end_date)

# 모든 데이터프레임을 하나의 데이터프레임으로 결합
df = pd.concat({k: v[['Open', 'High', 'Low', 'Close', 'Volume']] for k, v in stock_data.items()}, axis=1)
print(df.head())

# 데이터 스케일링 함수
def scale_data(df, method='standard'):
    """
    데이터를 스케일링합니다.

    Args:
    df (pd.DataFrame): 입력 데이터프레임
    method (str): 'minmax', 'standard', 'log' 중 하나 선택

    Returns:
    pd.DataFrame: 스케일링된 데이터프레임
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    elif method == 'standard':
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    elif method == 'log':
        scaled_df = np.log1p(df)  # log(1 + x) 변환을 사용하여 로그 변환
    else:
        scaled_df = df.copy()
    return scaled_df

# 기초 통계량 계산 함수
def calculate_basic_statistics(df):
    """
    주어진 데이터프레임의 기초 통계량을 계산합니다.

    Args:
    df (pd.DataFrame): 입력 데이터프레임

    Returns:
    pd.DataFrame: 기초 통계량을 포함하는 데이터프레임
    """
    daily_returns = df.pct_change().dropna()

    statistics = {
        'mean': df.mean(),
        'median': df.median(),
        'mode': df.apply(lambda x: mode(x).mode[0]),
        'variance': df.var(),
        'std_dev': df.std(),
        'min': df.min(),
        'max': df.max(),
        '25%': df.quantile(0.25),
        '50%': df.quantile(0.5),  # median과 동일
        '75%': df.quantile(0.75),
        'skewness': df.apply(lambda x: skew(x)),
        'kurtosis': df.apply(lambda x: kurtosis(x)),
        # 'cv': df.std() / df.mean(),  # 변동 계수
        # 'daily_mean_return': daily_returns.mean(),
        # 'daily_std_dev_return': daily_returns.std()
    }

    stats_df = pd.DataFrame(statistics)
    return stats_df

# 데이터 스케일링 (예: standard 스케일링)
scaled_df = scale_data(df, method='log')

scaled_df.columns = ['{}_{}'.format(*col) for col in scaled_df.columns]

price_df = scaled_df.loc[:,df.columns.get_level_values(1) != 'Volume']
volume_df = scaled_df.loc[:,df.columns.get_level_values(1) == 'Volume']

# sns.jointplot(x='MSFT_Open', y='GOOGL_Open', data=price_df, kind='scatter')

# 각 종목의 close만 추출하여 pairplot 
tmp = price_df.loc[:, price_df.columns.str.contains('Close')]
sns.pairplot(tmp, kind='reg')



df_diff = df.pct_change().dropna()
df_diff.columns = ['{}_{}'.format(*col) for col in df_diff.columns]
price_df_diff = df_diff.loc[:,df.columns.get_level_values(1) != 'Volume']
volume_df_diff = df_diff.loc[:,df.columns.get_level_values(1) == 'Volume']

# 각 종목의 close만 추출하여 pairplot 
tmp = price_df_diff.loc[:, price_df_diff.columns.str.contains('Close')]
sns.pairplot(tmp, kind='reg')

# 모든 종목에 대한 기초 통계량 계산
stats_df = calculate_basic_statistics(scaled_df)
print(stats_df.head())

# 가격 관련 통계량과 거래량 통계량 분리
price_stats = stats_df.loc[df.columns.get_level_values(1) != 'Volume']
volume_stats = stats_df.loc[df.columns.get_level_values(1) == 'Volume']

# 가격 관련 통계량 히트맵 시각화
plt.figure(figsize=(20, 10))
sns.heatmap(price_stats.T, annot=False, cmap='coolwarm')
plt.title('Price Related Statistics Heatmap')
plt.show()

# 가격 관련 통계량 박스 플롯 시각화
plt.figure(figsize=(20, 10))
sns.boxplot(data=price_stats.T)
plt.title('Price Related Statistics Box Plot')
plt.xticks(rotation=90)
plt.show()

# 거래량 관련 통계량 히트맵 시각화
plt.figure(figsize=(20, 10))
sns.heatmap(volume_stats.T, annot=True, cmap='coolwarm')
plt.title('Volume Related Statistics Heatmap')
plt.show()

# 거래량 관련 통계량 박스 플롯 시각화
plt.figure(figsize=(20, 10))
sns.boxplot(data=volume_stats.T)
plt.title('Volume Related Statistics Box Plot')
plt.xticks(rotation=90)
plt.show()


# 바 차트 시각화 - 평균
plt.figure(figsize=(20, 10))
price_stats['mean'].plot(kind='bar')
plt.title('Mean of Prices for Each Stock')
plt.xlabel('Stock')
plt.ylabel('Mean')
plt.xticks(rotation=90)
plt.show()



# 히스토그램 - 가격 관련 통계량 분포
price_stats.T.hist(bins=30, figsize=(20, 15))
plt.suptitle('Histograms of Price Related Statistics')
plt.show()



