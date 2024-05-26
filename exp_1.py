# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 00:00:01 2022

@author: sapzzil
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'US_Stock_market'
file_nm = 'data_TSLA_daily.csv'
data_path = os.path.join(data_dir, file_nm)
data = pd.read_csv(data_path)

data = data.sort_values('date')
data.index = data.date
data = data[['4. close']]


# n일 단위로 나눈다
n = 5
div_data = [data[i:i+n].to_numpy() for i in range(len(data)-n)]

# pattern 찾기
# vector 간 거리 기반
# 거리는 유클리드 거리를 사용

dist = list()
for i in range(len(div_data)):
    for j in range(len(div_data)):
        if i == j: continue        
        # noramlize
        a = (div_data[i] - div_data[i].mean()) - div_data[i].std()
        b = (div_data[j] - div_data[j].mean()) - div_data[j].std()
        # vector 거리
        dist.append([i,j,np.linalg.norm (a - b)])

near = [x for x in dist if x[2] < 0.5]

a_val = 11
b_val = 233
a = (div_data[a_val] - div_data[a_val].mean()) - div_data[a_val].std()
b = (div_data[b_val] - div_data[b_val].mean()) - div_data[b_val].std()
plt.plot(range(len(a)), a)
plt.plot(range(len(b)), b)
