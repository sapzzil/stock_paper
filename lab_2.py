# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:43:17 2024

@author: sapzzil

chatGpt에서 샘플 코드 실행해보는 코드
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 주식 데이터 가져오기
def get_stock_data(ticker, period="2y"):
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

ticker = "NVDA"
data = get_stock_data(ticker)

# 이동 평균 계산
data['SMA_50'] = data['Close'].rolling(window=5).mean()
data['SMA_200'] = data['Close'].rolling(window=20).mean()

# 결측치 제거
data = data.dropna()

# 추세 탐지 및 라벨링
data['Trend'] = 0  # 0: No trend, 1: Uptrend, -1: Downtrend
for i in range(1, len(data)):
    if data['SMA_50'].iloc[i] > data['SMA_200'].iloc[i] and data['SMA_50'].iloc[i-1] <= data['SMA_200'].iloc[i-1]:
        data['Trend'].iloc[i] = 1  # 상승 추세 시작
    elif data['SMA_50'].iloc[i] < data['SMA_200'].iloc[i] and data['SMA_50'].iloc[i-1] >= data['SMA_200'].iloc[i-1]:
        data['Trend'].iloc[i] = -1  # 하락 추세 시작

# 추세 직전 데이터 추출 함수
def get_pre_trend_data(data, trend_value, window=20):
    pre_trend_data = []
    for i in range(window, len(data)):
        if data['Trend'].iloc[i] == trend_value:
            pre_trend_data.append(data.iloc[i-window:i])
    return pd.concat(pre_trend_data)

# 상승 추세 직전 데이터
pre_uptrend_data = get_pre_trend_data(data, trend_value=1, window=20)

# 하락 추세 직전 데이터
pre_downtrend_data = get_pre_trend_data(data, trend_value=-1, window=20)

# 기초 통계량 계산 함수
def calculate_statistics(data):
    return data.describe()

# 상승 추세 직전 기초 통계량
uptrend_stats = calculate_statistics(pre_uptrend_data)

# 하락 추세 직전 기초 통계량
downtrend_stats = calculate_statistics(pre_downtrend_data)

# 통계량 출력
print("상승 추세 직전 기초 통계량:")
print(uptrend_stats)

print("\n하락 추세 직전 기초 통계량:")
print(downtrend_stats)

# 상승 추세 직전 데이터 시각화
plt.figure(figsize=(14, 7))
plt.plot(pre_uptrend_data.index, pre_uptrend_data['Close'], label='Close Price before Uptrend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('NVIDIA Stock Price before Uptrend')
plt.legend()
plt.show()

# 하락 추세 직전 데이터 시각화
plt.figure(figsize=(14, 7))
plt.plot(pre_downtrend_data.index, pre_downtrend_data['Close'], label='Close Price before Downtrend', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('NVIDIA Stock Price before Downtrend')
plt.legend()
plt.show()
