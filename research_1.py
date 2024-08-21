# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 12:05:38 2024

@author: sapzzil

n일간의 데이터로 m일 후의 종가 상승/하락 을 예측하는 classification 진행

"""

import pandas as pd
import os
import numpy as np



def create_dataset(stock):
    
    pass

# In[]:
n = 5
m = 1

base_dir = 'stock_market'
file_list = os.listdir(base_dir)

for i, file_nm in enumerate(file_list):
    stock_nm = file_nm.split('.')[0]
    
    data_path = os.path.join(base_dir, file_nm)
    stock = pd.read_csv(data_path)
    stock = stock.sort_values('Date')
    stock.index = stock.Date
    stock = stock.drop(columns=['Date','Close'])
    stock = stock.rename(columns={'Adj Close':'Close'})
    # stock['Y'] = stock.Close.shift(-m)
    
    diff_data = stock[['Close']].pct_change(m).shift(-m)
    
    stock['Y'] = [1 if x> 0.004 else 0 for x in diff_data.values]
    
    stock = stock[:-m]
    
    # log 변환 / skew 처리
    # Volume은 log10을 적용, 나머지는 log2
    stock[['Open', 'High', 'Low', 'Close']] = np.log2(stock[['Open', 'High', 'Low', 'Close']])
    stock['Volume'] = np.log10(stock['Volume'])
    
    
    # train/test dataset 분리
    train_end_idx = int(len(stock) * 0.7)
    test_start_idx = train_end_idx + n
    train_dataset = stock[:train_end_idx]
    test_dataset = stock[test_start_idx:]
    
    create_dataset()
    
    
    
    break




























