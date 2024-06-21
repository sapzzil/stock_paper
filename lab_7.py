# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:31:20 2024

@author: sapzzil
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def add_ta(df):
    # 거래량  지표
    # money flow index
    df['mfi'] = ta.volume.money_flow_index(df.High, df.Low, df.Close, df.Volume)
    # on balance volume
    df['obv'] = ta.volume.on_balance_volume(df.Close, df.Volume)
    # volume price trend
    df['vpt'] = ta.volume.volume_price_trend(df.Close, df.Volume)
    
    
    # In[6]:
    
    
    # 추세지표
    # ADX(Average Directional Movement Index)
    df['adx'] = ta.trend.adx(df.High, df.Low, df.Close)
    # CCI(Commodity Channel Index)
    df['cci'] = ta.trend.cci(df.High, df.Low, df.Close)
    # MACD(Moving Average Convergence & Divergence)
    df['macd'] = ta.trend.macd(df.Close)
    # SMA(Simple Moving Average)
    df['sma'] = ta.trend.sma_indicator(df.Close, window=20)
    # WMA(Weighted Moving Average)
    df['wma'] = ta.trend.wma_indicator(df.Close)
    
    
    # In[7]:
    
    
    # 모멘텀 지표
    # RSI(Relative Strength Index)
    df['rsi'] = ta.momentum.rsi(df.Close)
    # Stochastic Oscillator
    df['stoch_osc'] = ta.momentum.stoch(df.High, df.Low, df.Close)
    # Stochastic RSI
    df['stoch_rsi'] = ta.momentum.stochrsi(df.Close)
    # Price ROC(Price Rate of Change)
    df['roc'] = ta.momentum.roc(df.Close)
    
    
    # In[8]:
    
    
    # 변동성 지표
    # Bollinger Bands
    ## high
    df['boll_high'] = ta.volatility.bollinger_hband(df.Close)
    ## low
    df['boll_low'] = ta.volatility.bollinger_lband(df.Close)
    ## moving average
    df['boll_ma'] = ta.volatility.bollinger_mavg(df.Close)
    ## width
    df['boll_width'] = ta.volatility.bollinger_wband(df.Close)
    
    # Keltner Channel
    ## high
    df['keltner_high'] = ta.volatility.keltner_channel_hband(df.High, df.Low, df.Close)
    ## low
    df['keltner_low'] = ta.volatility.keltner_channel_lband(df.High, df.Low, df.Close)
    ## middle/center
    df['keltner_middle'] = ta.volatility.keltner_channel_mband(df.High, df.Low, df.Close)
    ## width
    df['keltner_width'] = ta.volatility.keltner_channel_wband(df.High, df.Low, df.Close)
    
    
    # In[9]:
    
    
    # 기타 지표
    # CR(Cumulative Return)
    df['cr'] = ta.others.cumulative_return(df.Close)
    # DLR(Daily Log Return)
    df['dlr'] = ta.others.daily_log_return(df.Close)

    return df

# In[]


# 종목 리스트
stock_list = ['MSFT', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSM', 'JPM', 'TSLA', 'WMT']
start_date = "2020-01-01"
end_date = "2023-12-31"

# 데이터 가져오기
def fetch_stock_data(stock_nm, start, end):
    return yf.download(stock_nm, start=start, end=end)

stock_data = fetch_stock_data('NVDA', start_date, end_date)
stock_data = stock_data.drop(columns='Close')
stock_data = stock_data.rename(columns={'Adj Close':'Close'})

stock_data.loc[:, ~stock_data.columns.str.contains('Volume')].plot.kde()
stock_data.loc[:, stock_data.columns.str.contains('Volume')].plot.kde()

price_data = stock_data.loc[:, ~stock_data.columns.str.contains('Volume')].pct_change().dropna()
price_data.plot.kde()

volume_data = stock_data.loc[:, stock_data.columns.str.contains('Volume')].pct_change().dropna()
volume_data.plot.kde()

tech_data = add_ta(stock_data.copy())
tech_corr = tech_data.corr() # loc[:,tech_data.columns.difference(['Open','High','Low'])].corr()
sns.heatmap(tech_corr, annot = False, cmap = 'YlGnBu')
features = tech_corr.loc[:, tech_corr['Close'] < 0.7].columns

tech_data = tech_data.loc[:,features.tolist() + ['Close']].dropna()

tech_desc = tech_data.describe()


# 아래 지표는 standard scale // 음수가 나오는 지표들
standard_scale_features = ['cci', 'macd', 'stoch_osc', 'roc', 'dlr']
s_scaler = StandardScaler()
tech_data[standard_scale_features] = s_scaler.fit_transform(tech_data[standard_scale_features])


# 나머지는 minMax scale
mm_scaler = MinMaxScaler()
tech_data.loc[:, tech_data.columns.difference(standard_scale_features)] = mm_scaler.fit_transform(tech_data.loc[:, tech_data.columns.difference(standard_scale_features)])


for i,col in enumerate(tech_data.columns):
    plt.figure(i)
    tech_data.loc[:,col].plot.kde()



n_days = 5  # 사용할 기간
X_data = []
Y_data = []
for i in range(len(df) - n_days * 2 + 1):
    X = df.iloc[i:i+n_days].drop(columns=['Close'])
    Y = df.iloc[i+n_days:i+n_days*2]['Close'].values
    X_data.append(X.values)
    Y_data.append(Y)

train_data_len = int(len(X_data) * 0.8)
train_data = X_data[:train_data_len]
test_data = X_data[train_data_len+5:]

train_X = np.array(X_data[:train_data_len])
train_Y = np.array(Y_data[:train_data_len])
test_X = np.array(X_data[train_data_len:])
test_Y = np.array(Y_data[train_data_len:])


# 샘플 데이터 생성 (여기서는 임의의 데이터를 사용)
samples = 100  # 샘플 수
time_steps = train_X[0].shape[0]
input_dim = train_X[0].shape[1]


# LSTM 모델 구축
model = Sequential()
model.add(LSTM(150, return_sequences=True, input_shape=(time_steps, input_dim)))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(50))
model.add(Dense(25))
model.add(Dense(5))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 구조 확인
model.summary()


# 모델 학습
history = model.fit(train_X, train_Y, batch_size=samples, epochs=150)

# 예측 수행
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)


n = 10
tmp = pd.DataFrame(train_predict[n],columns=['train_pred'])
tmp['train_Y'] = train_Y[n]
tmp.plot()

n = 89
tmp = pd.DataFrame(test_predict[n],columns=['test_pred'])
tmp['test_Y'] = test_Y[n]
tmp.plot()
 
# 모든 데이터프레임을 하나의 데이터프레임으로 결합
df = add_ta(stock_data.copy())
df_desc = df.describe()


stock_data_diff = stock_data.pct_change().dropna()
df_diff = add_ta(stock_data_diff.copy())
df_diff_desc = df_diff.describe()






Money Flow Index (MFI)
Moving Average Convergence & Divergence (MACD)
Relative Strength Index (RSI)
Bollinger Bands
Daily Log Return (DLR)


















