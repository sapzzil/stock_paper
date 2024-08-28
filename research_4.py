# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:14:16 2024

@author: sapzzil

결과 분석
"""

import pandas as pd

cls_result = 'research_2_model_result.csv'
reg_result = 'research_3_model_result.csv'


result_2 = pd.read_csv(cls_result)
result_3 = pd.read_csv(reg_result)

for col in result_3.loc[:,result_3.columns.str.contains('R-squared')].columns:
    print(f'{col}')
    print(result_3.nlargest(10, col))