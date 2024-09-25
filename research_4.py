# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:14:16 2024

@author: sapzzil

결과 분석
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

cls_result = 'research_2_model_result.csv'
reg_result = 'research_3_model_result.csv'


result_2 = pd.read_csv(cls_result)
result_3 = pd.read_csv(reg_result)

total_result = pd.DataFrame()
for col in result_3.loc[:,result_3.columns.str.contains('R-squared')].columns:
    print(f'{col}')
    top = result_3.nlargest(10, col)
    total_result[col] = top.sort_values('Stock').Stock.values
    

# 데이터프레임 전체에서 가장 많이 나온 값 찾기
most_frequent_value = total_result.stack().mode()[0]
    
# 기본 디렉토리 설정
base_directory = 'stock_data'

ticker_nm = 'CTLT'
file_path = os.path.join(base_directory, f'{ticker_nm}.csv')

df = pd.read_csv(file_path)
Value = 'Adj Close'
df[Value].plot()



# 데이터를 train(70%)와 test(30%)로 나누기
train_size = int(len(df) * 0.7)
train = df[:train_size]
test = df[train_size:]

# 그래프 시각화
plt.figure(figsize=(10, 6))

# Train 데이터 (파란색)
plt.plot(train.index, train[Value], label='Train Data (70%)', color='blue')

# Test 데이터 (빨간색)
plt.plot(test.index, test[Value], label='Test Data (30%)', color='red')

# 제목과 라벨 추가
plt.title('Train vs Test Data')
plt.xlabel('Date')
plt.ylabel(Value)

# 범례 추가
plt.legend()

# 그래프 출력
plt.show()
