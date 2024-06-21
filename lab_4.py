# -*- coding: utf-8 -*-
"""
Created on Mon May 27 09:36:09 2024

@author: sapzzil

기초통계 분석
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns


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

stocks = pd.DataFrame()
for i, file_nm in [(0,'data_AAPL_daily.csv')]: #enumerate(file_list): #
    stock_nm = file_nm.split('.')[0].split('_')[1]
    stock = load_data(data_dir, file_nm)
    if len(stock) < 5000:
        continue
    stocks[stock_nm] = stock.Close[-3001:-1].values

# 표준화 (Standardization)
scaler = MinMaxScaler()
standardized_stocks = scaler.fit_transform(stocks)
standardized_df = pd.DataFrame(standardized_stocks, columns=stocks.columns, index=stocks.index)

# 표준화된 데이터의 기초 통계량 계산
standardized_stats = standardized_df.describe().T
standardized_stats['median'] = standardized_df.median()
standardized_stats = standardized_stats[['mean', 'median', 'std', 'min', 'max']]
# standardized_stats.columns = ['Mean', 'Median', 'Standard Deviation', 'Min', 'Max']


plt.figure(figsize=(14, 7))
plt.bar(x= standardized_stats.index, height=standardized_stats['std'])
plt.title('Standardized Stock Prices')
plt.legend(standardized_stats.columns[2])




# 히트맵을 사용하여 통계량 비교
plt.figure(figsize=(10, 8))
sns.heatmap(standardized_stats, annot=False, cmap='coolwarm')
plt.title('Standardized Stock Statistics Heatmap')
plt.show()
