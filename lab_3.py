# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:04:35 2024

@author: sapzzil

1. 기술지표 추가
2. correlation 분석 -> 서로 상관성 없는 데이터만 추출(ranking)
3. input : 10일, output : 5일 구성
4. lstm 모델로 예측 진행

"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import ta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

data_dir = 'US_Stock_market'
file_list = [f for f in os.listdir(data_dir) if f.split('_')[-1] != 'intraday.csv']

def load_data(data_dir, file_nm):
    data_path = os.path.join(data_dir, file_nm)
    data = pd.read_csv(data_path,  index_col='date')
    # if len(data) < 120:
    #     return None
    data = data.sort_values('date')
    data.columns = ['Open','High','Low','Close','Volume']
    
    return data

def add_technical_indicators(data):
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA30'] = data['Close'].rolling(window=30).mean()
    data['RSI'] = compute_rsi(data['Close'], 14)
    data['UpperBB'], data['LowerBB'] = compute_bollinger_bands(data['Close'], 20)
    return data

def compute_rsi(series, period):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, period):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

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

# 시계열 데이터 생성 함수
def create_dataset(dataset, time_step=10, y_cnt = 5):
    X, Y = [], []
    for i in range(len(dataset)-(time_step + y_cnt +1)):
        scaled_data = dataset[i:((i+time_step + y_cnt))] / (dataset[i] + 0.000000001)
        
        X.append(scaled_data[0:time_step])
        Y.append(scaled_data[time_step:time_step + y_cnt, -1])  # 'Close' 값을 예측 목표로 사용
    return np.array(X), np.array(Y)


def cal_correlation(df, method='pearson'):
    return df.corr(method=method).astype('float32')
    

def feature_selection(df, corr_x_quantile = 0.7):#, corr_y_quantile = 0.85):

    corr_result = cal_correlation(df)
    
    # corr_result = corr_result[(corr_result.Y > corr_result.Y.quantile(corr_y_quantile)) | (corr_result.Y < corr_result.Y.quantile(1- corr_y_quantile))]
    # corr_result = corr_result[corr_result.index]
    
    # corr_coef = corr_result.quantile(corr_x_quantile).max()
    pos_corr_coef = corr_result.quantile(corr_x_quantile)
    neg_corr_coef = corr_result.quantile(1-corr_x_quantile)
    
    check_list = pd.DataFrame()

    for col in [col for col in corr_result.index if col != 'Adj Close']:
        if col == 'Adj Close':
            continue
        tmp = corr_result.loc[(corr_result.index != 'Adj Close') & 
                              ((corr_result[col] >= pos_corr_coef) | 
                              (corr_result[col] < neg_corr_coef)), col]
        # tmp = corr_result.loc[(~corr_result.index.str.contains('Adj Close')) & 
        #                       (corr_result[col].abs() >= corr_coef), col]
        if len(tmp) <= 1 or tmp.index.values[-1] == col :
            continue
        tmp = pd.merge(tmp, corr_result['Adj Close'], left_index=True, right_index=True, how='left')
        tmp['feature'] = col
        tmp = tmp.sort_values(by='Adj Close', ascending=False)
        
        # if tmp.index.values[0] == col:
        #     tmp.loc[col, 'use'] = True
        #     tmp.loc[pd.isna(tmp.use),'use'] = False
        
        # else :
        tmp['feature_Y_coef'] = tmp['Adj Close'][col]
        tmp['use'] = tmp['feature_Y_coef'].abs() < tmp['Adj Close'].abs()
        tmp = tmp.rename(columns = {col:'feature_selected_coef'})

        check_list = pd.concat([check_list,tmp.loc[tmp.use,:]])
    check_list = check_list[['feature', 'feature_selected_coef',  'feature_Y_coef', 'Adj Close']]
    check_list['selected'] = check_list.index    

    # check_list.loc[(check_list.correlation_coef == 1) & ()]

    check_feature = check_list.feature.unique()
    check_selected = check_list.selected.unique()
    # tmp = check_list.groupby('selected').count()
    
    # selected = [x for x in check_selected if x not in check_feature]
    selected = [x for x in corr_result.index if x not in check_feature]
    selected = [x for x in selected if x in check_selected]
    # selected_list = check_list[check_list.index.isin(selected)]

    return selected

start_date = "2020-01-01"
end_date = "2023-12-31"

for i, file_nm in [(0,'data_AAPL_daily.csv')]:#enumerate(file_list):
    stock_nm = file_nm.split('.')[0]
    # stock = load_data(data_dir, file_nm)    
    stock = yf.download('AAPL', start=start_date, end=end_date)
    # stock = add_technical_indicators(stock)
    stock = add_ta(stock)
    stock = stock[30:]
    stock = stock.drop(columns=['Close'])
    
    selected = feature_selection(stock,0.9)    
    features = selected + ['Adj Close']
    stock = stock[features]
    stock = stock.dropna().values
    
    

    # 시계열 데이터 생성
    time_step = 20
    X, Y = create_dataset(stock, time_step)
    
    # 학습 데이터와 테스트 데이터로 분리
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size
    X_train, X_test = X[0:train_size], X[train_size +  2* time_step:len(X)]
    Y_train, Y_test = Y[0:train_size,], Y[train_size +  2* time_step:len(Y)]
    
    # 데이터 형태 확인
    print("X_train shape:", X_train.shape)
    print("Y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)



    
    
    # LSTM 모델 구축
    model = Sequential()
    model.add(LSTM(150, return_sequences=True, input_shape=(time_step, len(features))))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(50))
    model.add(Dense(25))
    model.add(Dense(5))
    
    # 모델 컴파일
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # 모델 구조 확인
    model.summary()


    # 모델 학습
    history = model.fit(X_train, Y_train, batch_size=100, epochs=150)

    # 예측 수행
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    
    n = 1
    tmp = pd.DataFrame(train_predict[n],columns=['train_pred'])
    tmp['Y_train'] = Y_train[n]
    tmp.plot()
    
    n = 89
    tmp = pd.DataFrame(test_predict[n],columns=['test_pred'])
    tmp['Y_test'] = Y_test[n]
    tmp.plot()
    
    # 데이터 역정규화
    # train_predict = scaler.inverse_transform(np.concatenate((train_predict, np.zeros((train_predict.shape[0], len(features)-1))), axis=1))[:,0]
    # test_predict = scaler.inverse_transform(np.concatenate((test_predict, np.zeros((test_predict.shape[0], len(features)-1))), axis=1))[:,0]
    # Y_train = scaler.inverse_transform(np.concatenate((Y_train.reshape(-1, 1), np.zeros((Y_train.shape[0], len(features)-1))), axis=1))[:,0]
    # Y_test = scaler.inverse_transform(np.concatenate((Y_test.reshape(-1, 1), np.zeros((Y_test.shape[0], len(features)-1))), axis=1))[:,0]
    
    # 실제 데이터와 예측 데이터 시각화
    
    
    # plt.figure(figsize=(14, 7))
    # plt.plot(Y_train, label='Actual Train Data')
    # plt.plot(train_predict, label='Train Predict')
    # plt.plot(range(len(Y_train), len(Y_train) + len(Y_test)), Y_test, label='Actual Test Data')
    # plt.plot(range(len(Y_train), len(Y_train) + len(Y_test)), test_predict, label='Test Predict')
    # plt.xlabel('Date')
    # plt.ylabel('Price')
    # plt.title('NVIDIA Stock Price Prediction')
    # plt.legend()
    # plt.show()
















