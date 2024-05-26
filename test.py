# 2023-01-01 부터 2023-01-31 까지의 삼성전자 종가 데이터프레임
from pandas_datareader import data
import pandas as pd

start_date = '2023-01-01'
end_date = '2023-01-31' 
panel_data = data.DataReader('005930.KS', 'yahoo', start_date, end_date)