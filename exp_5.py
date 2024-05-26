# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 23:48:50 2023

@author: sapzzil
"""

import pandas as pd
import os
import numpy as np
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


########################################################################################################################
########################################################################################################################

def print_evaluation_metrics(name, pred, Y_test):
    acc = 100*accuracy_score(Y_test, pred)
    recall = 100*recall_score(Y_test, pred)
    precision = 100*precision_score(Y_test, pred)
    f1 = 100*f1_score(Y_test, pred)
    auc = 100 * roc_auc_score(Y_test, pred)
    print("{} model Accuracy = {:.2f}%".format(name, acc))
    print("{} model Recall = {:.2f}%".format(name, recall))
    print("{} model Precision = {:.2f}%".format(name, precision))
    print("{} model f1_score = {:.2f}%".format(name, f1))
    print("{} model AUC = {:.2f}%".format(name, auc))
    return acc, recall, precision, f1, auc


def strategy_1(result):
    '''
    pred가 1 일때, 무조건 삼
    마지막 종가 와 평균 비용의 차이를 구함
    

    Parameters
    ----------
    result : pd.DataFrame

    Returns
    -------
    pred_profit : TYPE
        DESCRIPTION.
    real_profit : TYPE
        DESCRIPTION.

    '''
    buy_cnt = 1 # 기본적으로 한 주씩만 사는거로 한다
    
    pred_cost = 0
    pred_cnt = 0
    real_cost = 0
    real_cnt = 0
    
    for price, pct, real_y, pred in result.values:
        # print(price, pct, real_y, pred)
        if pred == 1:
            pred_cnt += buy_cnt
            pred_cost += price * buy_cnt
        
        if real_y == 1:
            real_cnt += buy_cnt
            real_cost += price * buy_cnt
            
    avg_pred_cost = pred_cost / pred_cnt
    avg_real_cost = real_cost / real_cnt
    pred_profit = (price - avg_pred_cost) * pred_cnt
    real_profit = (price - avg_real_cost) * real_cnt
    return pred_profit, real_profit



def strategy_2(result, predict_day):
    '''
    pred가 1 일때, 무조건 삼
    predict_day 후 판매
    차액들의 합을 결과 
    

    Parameters
    ----------
    result : pd.DataFrame

    Returns
    -------
    pred_profit : TYPE
        DESCRIPTION.
    real_profit : TYPE
        DESCRIPTION.

    '''
    
    pred_prev_prices = list()
    pred_day_cnt = list()
    pred_profit = 0
    
    real_prev_prices = list()
    real_day_cnt = list()
    real_profit = 0
    
    for i, (price, pct, real_y, pred) in enumerate(result.values):
        if len(pred_day_cnt) != 0:
            pred_day_cnt = [c + 1 for c in pred_day_cnt]                
            if pred_day_cnt[0] == predict_day:
                pred_day_cnt.pop(0)
                pred_profit += price - pred_prev_prices.pop(0)
                
        if pred == 1:                
            pred_prev_prices.append(price)
            pred_day_cnt.append(0)
        
        ############################################################
        if len(real_day_cnt) != 0:
            real_day_cnt = [c + 1 for c in real_day_cnt]
            if real_day_cnt[0] == predict_day:
                real_day_cnt.pop(0)
                real_profit += price - real_prev_prices.pop(0)
        if real_y == 1:                
            real_prev_prices.append(price)
            real_day_cnt.append(0)
            
    return pred_profit, real_profit


def strategy_3(result):
    '''
    pred가 1 일때, 무조건 삼
    pred가 0 일때, 사둔것들 다 팜
    

    Parameters
    ----------
    result : pd.DataFrame

    Returns
    -------
    pred_profit : TYPE
        DESCRIPTION.
    real_profit : TYPE
        DESCRIPTION.

    '''
    buy_cnt = 1 # 기본적으로 한 주씩만 사는거로 한다
    pred_cost = 0
    pred_cnt = 0
    pred_profit = 0
    real_cost = 0
    real_cnt = 0
    real_profit = 0
    
    for price, pct, real_y, pred in result.values:
        # print(price, pct, real_y, pred)
        if pred == 1:
            pred_cnt += buy_cnt
            pred_cost += price * buy_cnt
        
        elif pred == 0 and pred_cnt != 0:
            pred_profit += (price - (pred_cost / pred_cnt)) * pred_cnt
            pred_cost = 0
            pred_cnt = 0
       
        if real_y == 1:
            real_cnt += buy_cnt
            real_cost += price * buy_cnt
        
        elif real_y == 0 and real_cnt != 0:
            real_profit += (price - (real_cost / real_cnt)) * real_cnt
            real_cost = 0
            real_cnt = 0
            

    return pred_profit, real_profit

def get_result(model, x_train, y_train, x_test, y_test, real, name = 'model'):        
    model.fit(x_train, y_train)
    
    pred = model.predict(x_test)
    
    acc = print_evaluation_metrics(name, pred, y_test)
    
    result = real.copy()
    result['pred'] = pred.copy()

    return acc, result



########################################################################################################################
########################################################################################################################

# data set 구성
# n일 단위로 나눠서 2일 후의 상승/하락을 맞춘다
n = 5
predict_day = 2
data_dir = 'US_Stock_market'
data_cols = ['n-4','n-3','n-2','n-1','n-0','Y']
result_cols = ['name', 'total_len','train_len','test_len',
               'logistic_regression_accuracy', 'logistic_regression_recall', 'logistic_regression_precision', 'logistic_regression_f1_score', 'logistic_regression_AUC', 'logistic_regression_strategy1_pred_profit','logistic_regression_strategy1_real_profit','logistic_regression_strategy2_pred_profit','logistic_regression_strategy2_real_profit','logistic_regression_strategy3_pred_profit','logistic_regression_strategy3_real_profit',
               'svm_linear_accuracy', 'svm_linear_recall', 'svm_linear_precision', 'svm_linear_f1_score', 'svm_linear_AUC', 'svm_linear_regression_strategy1_pred_profit','svm_linear_regression_strategy1_real_profit','svm_linear_regression_strategy2_pred_profit','svm_linear_regression_strategy2_real_profit','svm_linear_regression_strategy3_pred_profit','svm_linear_regression_strategy3_real_profit',
               'svc_rbf_accuracy', 'svc_rbf_recall', 'svc_rbf_precision', 'svc_rbf_f1_score', 'svc_rbf_AUC', 'svc_rbf_regression_strategy1_pred_profit','svc_rbf_regression_strategy1_real_profit','svc_rbf_regression_strategy2_pred_profit','svc_rbf_regression_strategy2_real_profit','svc_rbf_regression_strategy3_pred_profit','svc_rbf_regression_strategy3_real_profit',
               'knn_accuracy', 'knn_recall', 'knn_precision', 'knn_f1_score', 'knn_AUC', 'knn_regression_strategy1_pred_profit','knn_regression_strategy1_real_profit','knn_regression_strategy2_pred_profit','knn_regression_strategy2_real_profit','knn_regression_strategy3_pred_profit','knn_regression_strategy3_real_profit',
               'dtree_accuracy', 'dtree_recall', 'dtree_precision', 'dtree_f1_score', 'dtree_AUC', 'dtree_regression_strategy1_pred_profit','dtree_regression_strategy1_real_profit','dtree_regression_strategy2_pred_profit','dtree_regression_strategy2_real_profit','dtree_regression_strategy3_pred_profit','dtree_regression_strategy3_real_profit',
               'rforest_accuracy', 'rforest_recall', 'rforest_precision', 'rforest_f1_score', 'rforest_AUC', 'rforest_regression_strategy1_pred_profit','rforest_regression_strategy1_real_profit','rforest_regression_strategy2_pred_profit','rforest_regression_strategy2_real_profit','rforest_regression_strategy3_pred_profit','rforest_regression_strategy3_real_profit',
               'NN_accuracy', 'NN_recall', 'NN_precision', 'NN_f1_score', 'NN_AUC', 'NN_regression_strategy1_pred_profit','NN_regression_strategy1_real_profit','NN_regression_strategy2_pred_profit','NN_regression_strategy2_real_profit','NN_regression_strategy3_pred_profit','NN_regression_strategy3_real_profit',
               'xgboost_accuracy', 'xgboost_recall', 'xgboost_precision', 'xgboost_f1_score', 'xgboost_AUC', 'xgboost_regression_strategy1_pred_profit','xgboost_regression_strategy1_real_profit','xgboost_regression_strategy2_pred_profit','xgboost_regression_strategy2_real_profit','xgboost_regression_strategy3_pred_profit','xgboost_regression_strategy3_real_profit',
               'catboost_accuracy', 'catboost_recall', 'catboost_precision', 'catboost_f1_score', 'catboost_AUC', 'catboost_regression_strategy1_pred_profit','catboost_regression_strategy1_real_profit','catboost_regression_strategy2_pred_profit','catboost_regression_strategy2_real_profit','catboost_regression_strategy3_pred_profit','catboost_regression_strategy3_real_profit', ]


save_dir = 'results'
idx = 8
save_file_nm = f'result_{idx}'
save_file_path = os.path.join(save_dir, save_file_nm)
results = list()
skipped = list()
shap_lists = list()
file_list = [f for f in os.listdir(data_dir) if f.split('_')[-1] != 'intraday.csv']

if os.path.isfile('{}.csv'.format(save_file_path)):
    check = pd.read_csv('{}.csv'.format(save_file_path))
else:
    check = pd.DataFrame(columns = result_cols)
    check.to_csv('{}.csv'.format(save_file_path), header=True, index=False)

for i, file_nm in enumerate(file_list):
    stock_nm = file_nm.split('.')[0]
    if stock_nm in check.name.unique(): continue

    data_path = os.path.join(data_dir, file_nm)
    data = pd.read_csv(data_path)
    if len(data) < 120:
        continue
    data = data.sort_values('date')
    data.index = data.date
    data = data[['4. close']]
    # diff_data = data.pct_change(predict_day).shift(-predict_day)[:-predict_day]
    data['pct_chg'] = data.pct_change(predict_day).shift(-predict_day)
    data = data.dropna()
    
    # Y labeling
    # 미국 거래 수수료를 기준으로 계산이 필요
    # 환전 스프레드, 증권사 수수료, 환율 등을 고려해야하는데,
    # 여기선 일괄적으로 0.4%를 적용하겠다
    # 추후 고려가 필요한 부분
    label_values = 0.4 / 100
    # Y = [1 if x >= label_values else 0 for x in data.diff.values]
    # data = data[:-predict_day]
    data['Y'] = [1 if x >= label_values else 0 for x in data.pct_chg.values]
    
    
    dataset = [[data.to_numpy()[i:i+n,0], data.to_numpy()[i+n-1,-1]] for i in range(len(data) - n+1)]
    dataset = np.array(dataset)
    dataset = [x.tolist()+[y] for x,y in dataset]
    dataset = pd.DataFrame(dataset, columns=data_cols)
    
    data = data.iloc[n-1:,:]
    dataset.index = data.index
    
    ## normalize
    # 가로 방향으로 normalize
    # 전체를 기준으로 normalize를 하면 
    # 데이터 셋 전체에 대해서 mean, std를 구하기 때문에 cheating이 발생한다
    dataset.iloc[:,:-1] = dataset.iloc[:,:-1].subtract(dataset.iloc[:,:-1].mean(axis=1), axis=0).div(dataset.iloc[:,:-1].std(axis=1), axis=0)
    
     # 위의 방법으로 normalize를 진행하면
    # n일 간 계속 가격이 같을 시 std = 0  이고, inf값이 나온다
    # nan값을 drop해줄 필요가 있다
    # data 에서도 해당 부분을 drop해줘야 한다
    data = data.loc[dataset[dataset.notnull().all(axis=1)].index, :]
    dataset = dataset.dropna()
    
    # data split
    # train 70%, test 30%
    # 중간에 최소 n일(5일)의 간격을 두고 분리
    train_end_idx = int(len(dataset) * 0.8)
    test_start_idx = train_end_idx + n
    train_dataset = dataset[:train_end_idx]
    test_dataset = dataset[test_start_idx:]
    real = data[test_start_idx:]
   
    if len(train_dataset) < 120:
        skipped.append(file_nm)
        continue
    
    # train, test 에서 class가 한쪽에 치우치지 않게 수를 조절한다
    # inbalanced 문제를 해결하기 위함
    # class 중 적은 수를 구하고 그와 동일하게 class 각각의 개수를 맞춘다
    # data 중 선택을 하는건 random하게 선택한다
    # test는 class 개수를 맞출 필요가 없다
    train_even_cnt = min(len([x for x in train_dataset.Y if x == 1]), len([x for x in train_dataset.Y if x == 0]))
    # test_even_cnt = min(len([x for x in test_dataset.Y if x == 1]), len([x for x in test_dataset.Y if x == 0]))
    
    if train_even_cnt == 0 :
        skipped.append(file_nm)
        continue
    
    tmp = train_dataset[train_dataset.Y ==1].sample(train_even_cnt)
    tmp2 = train_dataset[train_dataset.Y ==0].sample(train_even_cnt)
    train_dataset = pd.concat([tmp,tmp2])
    
    # tmp = test_dataset[test_dataset.Y ==1].sample(test_even_cnt)
    # tmp2 = test_dataset[test_dataset.Y ==0].sample(test_even_cnt)
    # test_dataset = pd.concat([tmp,tmp2])
    
    # X, Y 분리    
    X_train = train_dataset.iloc[:,:-1]
    Y_train = train_dataset.iloc[:,-1]
    X_test = test_dataset.iloc[:,:-1]
    Y_test = test_dataset.iloc[:,-1]
    
    if len(X_train) < 120 or len(X_test) < 60:
        skipped.append(file_nm)
        continue

    
    result = list()
    # stock name
    result.append(stock_nm)
    
    # total_len
    result.append(len(dataset))
    # train_len
    result.append(len(X_train))
    # test_len
    result.append(len(X_test))

    
    ################################################################################

    model_name = 'logistic_regression'
    acc, tmp = get_result(LogisticRegression(), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'svm_linear'
    acc, tmp = get_result(SVC(kernel='linear', probability=True), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'svc_rbf'
    acc, tmp = get_result(SVC(kernel='poly'), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'knn'
    acc, tmp = get_result(KNeighborsClassifier(), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'dtree'
    acc, tmp = get_result(DecisionTreeClassifier(), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'rforest'
    acc, tmp = get_result(RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'nn'
    acc, tmp = get_result(MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'xgboost'
    acc, tmp = get_result(XGBClassifier(), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    model_name = 'catboost'
    acc, tmp = get_result(CatBoostClassifier(verbose=False), X_train, Y_train, X_test, Y_test, real, model_name)
    result.extend(acc)
    result.extend(strategy_1(tmp))
    result.extend(strategy_2(tmp, predict_day))
    result.extend(strategy_3(tmp))
    
    
    
    
    results.append(result)
    if i % 100 == 0:
        results = pd.DataFrame(results, columns = result_cols)
        results.to_csv("{}.csv".format(save_file_path), mode='a', header=False, index = False)
        results = list()
    
    
results = pd.DataFrame(results, columns = result_cols)
results.to_csv("{}.csv".format(save_file_path), mode='a', header=False, index = False)   
    
    
    
































