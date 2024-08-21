# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:18:02 2024

@author: sapzzil

regression
"""


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

# 기본 디렉토리 설정
base_directory = 'stock_data'

# 데이터 전처리 함수
def preprocess_data(df):
    # Date 컬럼을 인덱스로 설정하고, 원래 컬럼은 삭제
    df.set_index('Date', inplace=True)
    
    # Close 컬럼을 제거하고 Adj Close를 Close로 이름 변경
    df.drop(columns=['Close'], inplace=True)
    df.rename(columns={'Adj Close': 'Close'}, inplace=True)
    return df

# Train/Test 데이터 분할 (시간 순서 유지)
def split_train_test(df, window_size= 5, test_size=0.3):
    """
    데이터프레임을 주어진 비율로 train/test로 나누는 함수.
    
    Parameters:
    df (pandas.DataFrame): 입력 데이터프레임.
    test_size (float): test 데이터의 비율 (기본값은 0.3, 즉 30%).
    
    Returns:
    train_df (pandas.DataFrame): Train 데이터프레임.
    test_df (pandas.DataFrame): Test 데이터프레임.
    """
    
    # test_size를 기준으로 분할 인덱스 계산
    split_index = int(len(df) * (1 - test_size))
    
    # train과 test로 분할
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index + window_size:]
    
    return train_df, test_df

def transform_data(df, scaler=None, is_train=True, diff_order=1):
    """
    주식 데이터에 대해 로그 변환, 차분, 표준화를 순차적으로 적용하는 함수.
    Train 데이터와 Test 데이터를 다르게 처리합니다.
    
    Parameters:
    df (pandas.DataFrame): 원본 데이터프레임. 'Open', 'High', 'Low', 'Close', 'Volume' 컬럼이 포함되어 있어야 함.
    scaler (StandardScaler, optional): Train 데이터에서 학습된 스케일러. Test 데이터에 동일한 스케일링을 적용할 때 사용.
    is_train (bool): Train 데이터인지 여부를 지정. 기본값은 True.
    diff_order (int): 차분의 차수. 기본값은 1차 차분.
    
    Returns:
    df_transformed (pandas.DataFrame): 로그 변환, 차분, 표준화가 적용된 데이터프레임.
    scaler (StandardScaler): 표준화를 적용한 스케일러 객체 (Train 데이터의 경우 반환).
    last_values (dict): 원래 스케일로 복원하기 위해 필요한 마지막 원본 데이터 값들 (차분 복원을 위해).
    """

    # 1. 로그 변환
    # 1. 로그 변환 (0인 경우는 0으로 유지, 0이 아닌 경우만 로그 변환)
    df['Open'] = np.where(df['Open'] == 0, 0, np.log2(df['Open']))
    df['High'] = np.where(df['High'] == 0, 0, np.log2(df['High']))
    df['Low'] = np.where(df['Low'] == 0, 0, np.log2(df['Low']))
    df['Close'] = np.where(df['Close'] == 0, 0, np.log2(df['Close']))
    df['Volume'] = np.where(df['Volume'] == 0, 0, np.log10(df['Volume']))
    
    # 2. 차분 적용 (트렌드 제거)
    df_diff = df.diff(periods=diff_order).dropna()
    
    if is_train:
        # 차분 복원을 위해 필요한 마지막 원본 값 저장 (Train 데이터만 해당)
        last_values = df.iloc[-diff_order]
        
        # 3. 표준화 적용 (Train 데이터)
        scaler = StandardScaler()
        df_transformed = df_diff.copy()
        df_transformed[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df_diff[['Open', 'High', 'Low', 'Close', 'Volume']])
        
        return df_transformed, scaler, last_values
    
    else:
        # 차분 복원을 위해 필요한 마지막 원본 값 저장 (Test 데이터만 해당)
        last_values = df.iloc[-diff_order]
        
        # 3. 표준화 적용 (Test 데이터)
        df_transformed = df_diff.copy()
        df_transformed[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.transform(df_diff[['Open', 'High', 'Low', 'Close', 'Volume']])
        
        return df_transformed, last_values



def create_dataset(df, window_size=5, n=1):
    """
    회귀를 위한 X, y 데이터셋을 생성하는 함수.
    
    Parameters:
    df (pandas.DataFrame): 입력 데이터프레임. 로그 변환, 차분, 표준화된 데이터.
    m (int): X 데이터셋을 구성할 기간 (m일간의 피처 데이터).
    n (int): y 데이터셋을 구성할 예측 목표 기간 (n일 후의 Close 값).
    
    Returns:
    X (numpy.ndarray): 특징 데이터셋. 각 샘플은 (m * 피처 수) 형태의 벡터.
    y (numpy.ndarray): 타겟 데이터셋. 각 샘플은 n일 후의 Close 값.
    """
    
    X, y = [], []
    
    if len(df) <= window_size + n:
        raise ValueError("데이터가 충분하지 않습니다.")
    
    for i in range(len(df) - window_size - n + 1):
        # X 생성 (m일간의 데이터)
        X_window = df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[i:i + window_size].values.flatten()
        # y 생성 (n일 후의 Close 값)
        y_value = df['Close'].iloc[i + window_size + n - 1]
        
        # X와 y를 각각 리스트에 추가
        X.append(X_window)
        y.append(y_value)
    
    # X와 y를 각각 numpy 배열로 변환
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def reverse_transform(transformed_df, last_values, scaler):
    """
    변환된 데이터를 원래의 스케일로 복원하는 함수.
    regression일 때는 필요하다
    
    Parameters:
    transformed_df (pandas.DataFrame): 변환된 데이터프레임.
    last_values (pandas.Series): 차분 이전의 마지막 원본 값.
    scaler (StandardScaler): 표준화에 사용된 스케일러.
    
    Returns:
    restored_df (pandas.DataFrame): 복원된 원본 스케일의 데이터프레임.
    """
    
    # 1. 역표준화 (Inverse Standardization)
    restored_df = transformed_df.copy()
    restored_df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.inverse_transform(transformed_df[['Open', 'High', 'Low', 'Close', 'Volume']])
    
    # 2. 차분 복원 (Integration)
    for col in restored_df.columns:
        restored_df[col] = restored_df[col].cumsum() + last_values[col]
    
    # 3. 로그 변환 역변환 (Inverse Log Transformation)
    restored_df[['Open', 'High', 'Low', 'Close']] = np.exp2(restored_df[['Open', 'High', 'Low', 'Close']])
    restored_df['Volume'] = np.exp10(restored_df['Volume'])
    
    return restored_df


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_models(X_train, y_train, X_test, y_test):
    """
    여러 회귀 모델을 학습시키고, 성능을 평가한 후 결과를 반환하는 함수.
    
    Parameters:
    X_train (numpy.ndarray): 학습용 특징 데이터.
    y_train (numpy.ndarray): 학습용 타겟 데이터.
    X_test (numpy.ndarray): 테스트용 특징 데이터.
    y_test (numpy.ndarray): 테스트용 타겟 데이터.
    save_to_csv (bool): 결과를 CSV 파일로 저장할지 여부 (기본값은 False).
    csv_filename (str): CSV 파일명 (기본값은 "regression_model_results.csv").
    
    Returns:
    results_df (pandas.DataFrame): 각 모델의 성능 평가 결과가 담긴 데이터프레임.
    """
    
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cat

    # 모델 리스트 정의
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(),
        "SVR": SVR(),
        "XGBoost": xgb.XGBRegressor(),
        "LightGBM": lgb.LGBMRegressor(),
        "CatBoost": cat.CatBoostRegressor(verbose=0)
    }

    # 결과 저장용 딕셔너리
    results = {}

    # 각 모델에 대해 학습 및 평가
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # 예측
        y_pred = model.predict(X_test)
        
        # 평가 지표 계산
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # 결과 저장
        results[name] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R-squared": r2
        }

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results).T
    
    
    return results_df


all_results = []
# i = 0
for ticker in os.listdir(base_directory):
    ticker_nm = ticker.split('.')[0]
    file_path = os.path.join(base_directory, f'{ticker_nm}.csv')
    
    if os.path.exists(file_path):
        print(f"Processing data for {ticker_nm}...")
        df = pd.read_csv(file_path)
        if len(df) < 1000 : continue
        
        # 전처리 수행
        df = preprocess_data(df)
        train_df, test_df = split_train_test(df)
        
        train_df_transformed, scaler, last_values = transform_data(train_df)
        test_df_transformed, test_last_values = transform_data(test_df, scaler, is_train=False)
        
        train_X, train_y = create_dataset(train_df_transformed)
        test_X, test_y = create_dataset(test_df_transformed)
        
        
        results_df = evaluate_models(train_X, train_y, test_X, test_y)
        
        # 열 이름에 모델 이름을 포함하여 확장
        model_results_flattened = {}
        for model_name, metrics in results_df.iterrows():
            for metric_name, value in metrics.items():
                if metric_name != "Stock":  # "Stock" 컬럼이 없을 수도 있으므로 체크
                    new_col_name = f"{model_name}_{metric_name}"
                    model_results_flattened[new_col_name] = value
        # Stock 이름을 포함한 결과 추가
        model_results_flattened["Stock"] = ticker_nm
        all_results.append(model_results_flattened)
        print(f'{ticker_nm} is done')
    # if i == 5: break
    # i += 1
        
  
# 결과를 데이터프레임으로 변환
all_results_df = pd.DataFrame(all_results)

# 열 순서 정리 (Stock을 첫 번째 열로)
cols = ["Stock"] + [col for col in all_results_df.columns if col != "Stock"]
all_results_df = all_results_df[cols]     

# 현재 실행 중인 파이썬 파일 이름을 얻기
python_file_name = os.path.splitext(os.path.basename(__file__))[0]

# CSV 파일 이름 생성
csv_filename = f"{python_file_name}_model_result.csv"
all_results_df.to_csv(csv_filename, index=False)
print(f"All stock results saved to {csv_filename}")