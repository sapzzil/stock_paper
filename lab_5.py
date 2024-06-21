# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:59:59 2024

@author: sapzzil

다중공선성 제거
"""


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy.stats import mode, skew, kurtosis


def calculate_basic_statistics(df, scale_type='none'):
    """
    주어진 데이터프레임의 기초 통계량을 계산합니다. 
    선택적으로 데이터를 정규화하거나 표준화할 수 있습니다.

    Args:
    df (pd.DataFrame): 입력 데이터프레임
    scale_type (str): 'none', 'minmax', 'standard' 중 하나 선택

    Returns:
    pd.DataFrame: 기초 통계량을 포함하는 데이터프레임
    """
    # 데이터 스케일링
    if scale_type == 'minmax':
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    elif scale_type == 'standard':
        scaler = StandardScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    else:
        scaled_df = df.copy()
    
    # 일별 수익률 계산
    daily_returns = scaled_df.pct_change().dropna()

    # 기초 통계량 계산
    statistics = {
        'mean': scaled_df.mean(),
        'median': scaled_df.median(),
        'mode': scaled_df.apply(lambda x: mode(x).mode[0]),
        'variance': scaled_df.var(),
        'std_dev': scaled_df.std(),
        'min': scaled_df.min(),        
        '25%': scaled_df.quantile(0.25),
        '50%': scaled_df.quantile(0.5),  # median과 동일
        '75%': scaled_df.quantile(0.75),
        'max': scaled_df.max(),
        'IQR': scaled_df.quantile(0.75) - scaled_df.quantile(0.25),
        'skewness': scaled_df.skew(),
        'kurtosis': scaled_df.kurtosis(),
        'cv': scaled_df.std() / scaled_df.mean(),  # 변동 계수 계산
        'daily_mean_return': daily_returns.mean(),  # 일평균 수익률
        'daily_std_dev_return': daily_returns.std()  # 일평균 수익률의 표준 편차
    }
    
    stats_df = pd.DataFrame(statistics)
    return stats_df

def load_data_from_yfinance(tickers):
    data = pd.DataFrame()
    for ticker in tickers:
        # yf_data = yf.download(ticker, start="2020-01-01", end="2023-12-31")
        data[ticker] = yf.download(ticker, start="2020-01-01", end="2023-12-31")#['Adj Close']
    return data



stock_list = ['MSFT', 'AAPL','NVDA','GOOGL','AMZN','META','TSM','JPM','TSLA','WMT']
stock = load_data_from_yfinance(stock_list)


# scale
stock_scale = stock/stock.max()

# diff
stock_diff = stock.diff()[1:]

stock_desc = calculate_basic_statistics(stock)
stock_diff_desc = calculate_basic_statistics(stock_diff)



































