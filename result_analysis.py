#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 11:54:40 2023

@author: darby
"""


import pandas as pd
import os
import matplotlib.pyplot as plt

result_dir = 'results'
file_nm = 'result_9.csv'

file_path = os.path.join(result_dir, file_nm)

result = pd.read_csv(file_path)


tmp = result.loc[:, result.columns.str.contains('accuracy')].describe().T
tmp = result.loc[:, result.columns.str.contains('AUC')].describe().T
tmp = result.loc[:, (result.columns == 'name') | (result.columns.str.contains('strategy2_pred'))].describe().T


tmp2 = result.loc[:, (result.columns == 'name') | (result.columns.str.contains('logistic_regression')) & (result.columns.str.contains('pred'))]
tmp2.plot.box()



result.loc[:, result.columns.str.contains('strategy2_pred')].plot.kde()




data_dir = 'US_Stock_market'
data_file_nm = '{}.csv'.format('data_NTEI_daily')
data_file_path = os.path.join(data_dir, data_file_nm)

data = pd.read_csv(data_file_path)

data = data.sort_values('date')
data.index = data.date
data = data[['4. close']]
data.plot()