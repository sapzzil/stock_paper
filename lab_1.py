# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:04:35 2024

@author: sapzzil

가설 : 주가에는 미래 예측을 위한 시그널이 포함된다
실험
    1. 주가 데이터 정제 (신생이라 데이터 없는것 제외)
    2. 기술적 지표 계산
    3. feature extraction/selection
        1. 변수 상관관계 분석
        2. 변수 다중공선성 분석
        3. 
    4. model 학습 및 결과 도출
    5. 결과 활용 방안 실행
        1. 
"""

import pandas as pd
import os
import numpy as np
import ta
import matplotlib.pyplot as plt

data_dir = 'US_Stock_market'
file_list = [f for f in os.listdir(data_dir) if f.split('_')[-1] != 'intraday.csv']


# In[5]:

def add_ta(df):
    # 거래량  지표
    # money flow index
    df['mfi'] = ta.volume.money_flow_index(df.high, df.low, df.close, df.volume)
    # on balance volume
    df['obv'] = ta.volume.on_balance_volume(df.close, df.volume)
    # volume price trend
    df['vpt'] = ta.volume.volume_price_trend(df.close, df.volume)
    
    
    # In[6]:
    
    
    # 추세지표
    # ADX(Average Directional Movement Index)
    df['adx'] = ta.trend.adx(df.high, df.low, df.close)
    # CCI(Commodity Channel Index)
    df['cci'] = ta.trend.cci(df.high, df.low, df.close)
    # MACD(Moving Average Convergence & Divergence)
    df['macd'] = ta.trend.macd(df.close)
    # SMA(Simple Moving Average)
    df['sma'] = ta.trend.sma_indicator(df.close, window=20)
    # WMA(Weighted Moving Average)
    df['wma'] = ta.trend.wma_indicator(df.close)
    
    
    # In[7]:
    
    
    # 모멘텀 지표
    # RSI(Relative Strength Index)
    df['rsi'] = ta.momentum.rsi(df.close)
    # Stochastic Oscillator
    df['stoch_osc'] = ta.momentum.stoch(df.high, df.low, df.close)
    # Stochastic RSI
    df['stoch_rsi'] = ta.momentum.stochrsi(df.close)
    # Price ROC(Price Rate of Change)
    df['roc'] = ta.momentum.roc(df.close)
    
    
    # In[8]:
    
    
    # 변동성 지표
    # Bollinger Bands
    ## high
    df['boll_high'] = ta.volatility.bollinger_hband(df.close)
    ## low
    df['boll_low'] = ta.volatility.bollinger_lband(df.close)
    ## moving average
    df['boll_ma'] = ta.volatility.bollinger_mavg(df.close)
    ## width
    df['boll_width'] = ta.volatility.bollinger_wband(df.close)
    
    # Keltner Channel
    ## high
    df['keltner_high'] = ta.volatility.keltner_channel_hband(df.high, df.low, df.close)
    ## low
    df['keltner_low'] = ta.volatility.keltner_channel_lband(df.high, df.low, df.close)
    ## middle/center
    df['keltner_middle'] = ta.volatility.keltner_channel_mband(df.high, df.low, df.close)
    ## width
    df['keltner_width'] = ta.volatility.keltner_channel_wband(df.high, df.low, df.close)
    
    
    # In[9]:
    
    
    # 기타 지표
    # CR(Cumulative Return)
    df['cr'] = ta.others.cumulative_return(df.close)
    # DLR(Daily Log Return)
    df['dlr'] = ta.others.daily_log_return(df.close)

    return df

def div_stock(stock_data, period):
    return stock_data[-period:]

def calculate_basic_statistics(stock_data):
    """
    주어진 주식 데이터에 대한 기본 통계량을 계산합니다.
    :param stock_data: 주식 데이터 DataFrame
    :return: 기본 통계량 딕셔너리
    """
    close_prices = stock_data['close']
    daily_returns = close_prices.pct_change().dropna()

    stats = {
        'Mean': close_prices.mean(),
        'Median': close_prices.median(),
        'Standard Deviation': close_prices.std(),
        'Variance': close_prices.var(),
        'Min': close_prices.min(),
        'Max': close_prices.max(),
        'Range': close_prices.max() - close_prices.min(),
        'Q1': close_prices.quantile(0.25),
        'Q2 (Median)': close_prices.quantile(0.5),
        'Q3': close_prices.quantile(0.75),
        'Coefficient of Variation': close_prices.std() / close_prices.mean(),
        'Mean Daily Return': daily_returns.mean(),
        'Standard Deviation of Daily Return': daily_returns.std()
    }

    return stats

def calculate_periodic_statistics(stock):
    """
    주어진 주식 종목에 대해 다양한 기간별 통계량을 계산합니다.
    :param ticker: 주식 종목 티커
    :return: 기간별 통계량 딕셔너리
    """
    periods = {
        '3M': 50,
        '6M': 30,
        '1Y': 90
    }

    results = {}
    for period_name, period in periods.items():
        stock_data = div_stock(stock.copy(), period)
        stats = calculate_basic_statistics(stock_data)
        results[period_name] = stats

    return results

def calculate_rolling_statistics(stock_data, window=252):
    rolling_mean = stock_data['close'].rolling(window=window).mean()
    rolling_std = stock_data['close'].rolling(window=window).std()
    return rolling_mean, rolling_std

for i, file_nm in [(0,'data_AAPL_daily.csv')]:#enumerate(file_list):
    stock_nm = file_nm.split('.')[0]
    # if stock_nm in check.name.unique(): continue

    data_path = os.path.join(data_dir, file_nm)
    data = pd.read_csv(data_path,  index_col='date')
    if len(data) < 120:
        continue
    data = data.sort_values('date')
    data.columns = ['open','high','low','close','volume']
    
    # data_desc = data.describe()
    # ta.add_all_ta_features(data, open='open', high='high', low='low', close='close', volume='volume',fillna=True)
    
    # data_with_ta = add_ta(data.copy())
    
    periodic_stats = calculate_periodic_statistics(data.copy())

    # 결과 출력
    for period, stats in periodic_stats.items():
        print(f"--- {file_nm} - {period} 통계량 ---")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}")
        print()
        
    
    
    rolling_mean, rolling_std = calculate_rolling_statistics(data.copy())
    
    
    
    plt.figure(figsize=(14, 7))
    plt.plot(data['close'], label='Close Price')
    plt.plot(rolling_mean, label='1-Year Rolling Mean', linestyle='--')
    plt.plot(rolling_std, label='1-Year Rolling Std Dev', linestyle='--')
    plt.legend()
    plt.title(f'{file_nm} Rolling Statistics')
    plt.show()




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    