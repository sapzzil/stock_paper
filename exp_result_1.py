# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 19:35:39 2022

@author: sapzzil
"""
"""
import pandas as pd
import os
import matplotlib.pyplot as plt

save_dir = 'results'

result_shap = pd.DataFrame()
result = pd.DataFrame()
for file_nm in os.listdir(save_dir):
    file_path = os.path.join(save_dir,file_nm)
    if 'shap' in file_nm:
        tmp = pd.read_csv(file_path)
        result_shap = pd.concat([result_shap, tmp], axis=0)
    
    else:
        tmp = pd.read_csv(file_path)
        result = pd.concat([result, tmp], axis=0)

result = result[result.test_len >= 60]

result.groupby(['name']).mean()

result.iloc[:,4:].plot.kde()

shap_var = ['mean_n-4', 'mean_n-3', 'mean_n-2', 'mean_n-1', 'mean_n-0']
result_shap_abs = result_shap.copy()
result_shap_abs[shap_var] = result_shap_abs[shap_var].abs()

shap = result_shap_abs.groupby(['model']).mean()

for model, val in shap.iterrows():
    plt.figure()
    val[shap_var].plot(kind='barh')
    plt.title(model)
"""



import pandas as pd
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

result_dir = 'results'
shap_result = pd.DataFrame()
stock_result = pd.DataFrame()

for file_nm in os.listdir(result_dir):
    if not file_nm.endswith('csv'): continue
    
    file_path = os.path.join(result_dir,file_nm)
    
    if 'shap' in file_nm:
        tmp = pd.read_csv(file_path)
        shap_result = pd.concat([shap_result, tmp],axis=0).reset_index(drop=True)
    
    else:
        tmp = pd.read_csv(file_path)
        stock_result = pd.concat([stock_result, tmp],axis=0).reset_index(drop=True)
    

feature_list = ['mean_n-4', 'mean_n-3', 'mean_n-2', 'mean_n-1', 'mean_n-0']
shap_result.loc[:,feature_list] = shap_result.loc[:,feature_list].abs()


tmp = shap_result.groupby(['model']).mean()

for model in tmp.index:
    base_val = tmp.loc[model,'mean_base_values']
    tmp.loc[model,feature_list] += base_val
    plt.figure()
    tmp.loc[model,feature_list].plot(kind='barh')
    plt.title(model)
    plt.xlabel("shap values | base_value : {}".format(base_val))
    plt.ylabel("features")
    plt.axvline(x= base_val, color='r', linestyle='-', linewidth=3)
    plt.show()


tmp2 = stock_result.groupby(['name']).mean()    
tmp2 = tmp2.iloc[:,3:]
tmp2.plot.kde()
desc = tmp2.describe()
