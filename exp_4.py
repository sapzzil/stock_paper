# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 02:50:51 2022

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
import shap
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

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

# data set 구성
# n일 단위로 나눠서 2일 후의 상승/하락을 맞춘다
n = 5
predict_day = 2
data_dir = 'US_Stock_market'
data_cols = ['n-4','n-3','n-2','n-1','n-0','Y']
result_cols = ['name', 'total_len','train_len','test_len',
               'logistic_regression_accuracy', 'logistic_regression_recall', 'logistic_regression_precision', 'logistic_regression_f1_score', 'logistic_regression_AUC',
               'svm_linear_accuracy', 'svm_linear_recall', 'svm_linear_precision', 'svm_linear_f1_score', 'svm_linear_AUC',
               'svc_rbf_accuracy', 'svc_rbf_recall', 'svc_rbf_precision', 'svc_rbf_f1_score', 'svc_rbf_AUC',
               'knn_accuracy', 'knn_recall', 'knn_precision', 'knn_f1_score', 'knn_AUC',
               'dtree_accuracy', 'dtree_recall', 'dtree_precision', 'dtree_f1_score', 'dtree_AUC',
               'rforest_accuracy', 'rforest_recall', 'rforest_precision', 'rforest_f1_score', 'rforest_AUC',
               'NN_accuracy', 'NN_recall', 'NN_precision', 'NN_f1_score', 'NN_AUC',
               'xgboost_accuracy', 'xgboost_recall', 'xgboost_precision', 'xgboost_f1_score', 'xgboost_AUC',
               'catboost_accuracy', 'catboost_recall', 'catboost_precision', 'catboost_f1_score', 'catboost_AUC' ]

shap_cols = ['name','model','mean_n-4','mean_n-3','mean_n-2','mean_n-1','mean_n-0',
             'std_n-4','std_n-3','std_n-2','std_n-1','std_n-0',
             'mean_base_values','std_base_values']
save_dir = 'results'
idx = 8
save_file_nm = f'result_{idx}'
save_file_shap_nm = f'result_shap_{idx}'
save_file_path = os.path.join(save_dir, save_file_nm)
save_file_shap_path = os.path.join(save_dir, save_file_shap_nm)
results = list()
skipped = list()
shap_lists = list()
file_list = [f for f in os.listdir(data_dir) if f.split('_')[-1] != 'intraday.csv']
shap_cal = random.sample(file_list, int(len(file_list) * 0.05))

if os.path.isfile('{}.csv'.format(save_file_path)):
    check = pd.read_csv('{}.csv'.format(save_file_path))
else:
    check = pd.DataFrame(columns = result_cols)
    check.to_csv('{}.csv'.format(save_file_path), header=True, index=False)
    
if os.path.isfile('{}.csv'.format(save_file_shap_path)):
    check_shap = pd.read_csv('{}.csv'.format(save_file_shap_path))
else:
    check_shap = pd.DataFrame(columns = shap_cols)
    check_shap.to_csv('{}.csv'.format(save_file_shap_path), header=True, index=False)
    
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
    diff_data = data.pct_change(predict_day).shift(-predict_day)[:-2]
    
    # Y labeling
    # 미국 거래 수수료를 기준으로 계산이 필요
    # 환전 스프레드, 증권사 수수료, 환율 등을 고려해야하는데,
    # 여기선 일괄적으로 0.4%를 적용하겠다
    # 추후 고려가 필요한 부분
    label_values = 0.4 / 100
    Y = [1 if x >= label_values else 0 for x in diff_data.values]
    data = data[:-2]
    data['Y'] = Y
    dataset = [[data.to_numpy()[i:i+n,0], data.to_numpy()[i+n-1,1]] for i in range(len(data) - n)]
    dataset = np.array(dataset)
    dataset = [x.tolist()+[y] for x,y in dataset]
    dataset = pd.DataFrame(dataset, columns=data_cols)
    
    # data split
    # train 70%, test 30%
    # 중간에 최소 n일(5일)의 간격을 두고 분리
    train_end_idx = int(len(dataset) * 0.7)
    test_start_idx = train_end_idx + n
    train_dataset = dataset[:train_end_idx]
    test_dataset = dataset[test_start_idx:]
    
    ## normalize
    # 가로 방향으로 normalize
    # 전체를 기준으로 normalize를 하면 
    # 데이터 셋 전체에 대해서 mean, std를 구하기 때문에 cheating이 발생한다
    train_dataset.iloc[:,:-1] = train_dataset.iloc[:,:-1].subtract(train_dataset.iloc[:,:-1].mean(axis=1), axis=0).div(train_dataset.iloc[:,:-1].std(axis=1), axis=0)
    test_dataset.iloc[:,:-1] = test_dataset.iloc[:,:-1].subtract(test_dataset.iloc[:,:-1].mean(axis=1), axis=0).div(test_dataset.iloc[:,:-1].std(axis=1), axis=0)
    
    # 위의 방법으로 normalize를 진행하면
    # n일 간 계속 가격이 같을 시 std = 0  이고, inf값이 나온다
    # nan값을 drop해줄 필요가 있다
    train_dataset = train_dataset.dropna()
    test_dataset = test_dataset.dropna()
    
    if len(train_dataset) < 120:
        skipped.append(file_nm)
        continue
    
    # train, test 에서 class가 한쪽에 치우치지 않게 수를 조절한다
    # inbalanced 문제를 해결하기 위함
    # class 중 적은 수를 구하고 그와 동일하게 class 각각의 개수를 맞춘다
    # data 중 선택을 하는건 random하게 선택한다
    train_even_cnt = min(len([x for x in train_dataset.Y if x == 1]), len([x for x in train_dataset.Y if x == 0]))
    test_even_cnt = min(len([x for x in test_dataset.Y if x == 1]), len([x for x in test_dataset.Y if x == 0]))
    
    if train_even_cnt == 0 or test_even_cnt == 0:
        skipped.append(file_nm)
        continue
    
    tmp = train_dataset[train_dataset.Y ==1].sample(train_even_cnt)
    tmp2 = train_dataset[train_dataset.Y ==0].sample(train_even_cnt)
    train_dataset = pd.concat([tmp,tmp2])
    
    tmp = test_dataset[test_dataset.Y ==1].sample(test_even_cnt)
    tmp2 = test_dataset[test_dataset.Y ==0].sample(test_even_cnt)
    test_dataset = pd.concat([tmp,tmp2])
    
    # X, Y 분리    
    X_train = train_dataset.iloc[:,:-1]
    Y_train = train_dataset.iloc[:,-1]
    X_test = test_dataset.iloc[:,:-1]
    Y_test = test_dataset.iloc[:,-1]
    
    if len(X_train) < 120 or len(X_test) < 60:
        skipped.append(file_nm)
        continue

    shap_list = list()
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
    # logistic regression
    logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    acc = print_evaluation_metrics('logistic-regression' ,logistic.predict(X_test), Y_test)
    result.extend(acc)
    # shap.plots.waterfall(logistic_val[0])
    # logistic_explainer = shap.KernelExplainer(logistic.predict_proba, X_train)
    # logistic_val = logistic_explainer.shap_values(X_test)
    # shap.force_plot(logistic_explainer.expected_value[0], logistic_val[0], X_test)
    # shap.summary_plot(logistic_val, X_train, feature_names=X_train.columns)
    ################################################################################
    ################################################################################
    # Support Vector Machine Classifier with a linear kernel
    svm_linear = SVC(kernel='linear', probability=True)
    svm_linear.fit(X_train, Y_train)
    acc = print_evaluation_metrics('svm_linear' ,svm_linear.predict(X_test), Y_test)
    result.extend(acc)
    # svm_linear_explainer = shap.Explainer(svm_linear, X_train)
    # svm_linear_val = svm_linear_explainer(X_train)
    # shap.plots.waterfall(svm_linear_val[0])
    # svm_linear_explainer = shap.KernelExplainer(svm_linear.predict_proba, X_train)
    # svm_linear_val = svm_linear_explainer.shap_values(X_test)
    # shap.force_plot(svm_linear_explainer.expected_value[0], svm_linear_val[0], X_test)
    ################################################################################
    ################################################################################
    # Support Vector Machine Classifier with a radial basis function kernel
    svc_rbf = SVC(kernel='poly')
    svc_rbf.fit(X_train, Y_train)
    acc = print_evaluation_metrics('svc_rbf' ,svc_rbf.predict(X_test), Y_test)
    result.extend(acc)
    # shap.plots.waterfall(svc_rbf_val)
    # svc_rbf_explainer = shap.KernelExplainer(svc_rbf.predict_proba, X_train)
    # svc_rbf_val = svc_rbf_explainer.shap_values(X_test)
    # shap.force_plot(svc_rbf_explainer.expected_value[0], svc_rbf_val[0], X_test)
    ################################################################################
    ################################################################################
    # K-nearest neighbors classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    acc = print_evaluation_metrics('knn' ,knn.predict(X_test), Y_test)
    result.extend(acc)
    # shap.plots.waterfall(knn_val[0])
    # knn_explainer = shap.KernelExplainer(knn.predict_proba, X_train)
    # knn_val = knn_explainer.shap_values(X_test.iloc[0,:])
    # shap.force_plot(knn_explainer.expected_value[0], knn_val[0], X_test.iloc[0,:])
    ################################################################################
    ################################################################################
    # Decision Tree classifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, Y_train)
    acc = print_evaluation_metrics('dtree' ,dtree.predict(X_test), Y_test)
    result.extend(acc)
    # shap.plots.waterfall(dtree_val[0])
    # shap.force_plot(dtree_explainer.expected_value[0], dtree_val[0], X_test)
    ################################################################################
    ################################################################################
    # Random Forest classifier
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    rforest.fit(X_train, Y_train)
    acc = print_evaluation_metrics('rforest' ,rforest.predict(X_test), Y_test)
    result.extend(acc)
    # shap.force_plot(rforest_explainer.expected_value[0], rforest_val[0], X_test)
    ################################################################################
    ################################################################################
    # Multi Layer perceptron classifier
    nn = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
    nn.fit(X_train, Y_train)
    acc = print_evaluation_metrics('nn' ,nn.predict(X_test), Y_test)
    result.extend(acc)
    # shap.force_plot(nn_explainer.expected_value[0], nn_val[0], X_test)
    ################################################################################
    ################################################################################
    # XGBoost Classifier
    xgboost = XGBClassifier()
    xgboost.fit(X_train, Y_train)
    acc = print_evaluation_metrics('xgboost' ,xgboost.predict(X_test), Y_test)
    result.extend(acc)
    # shap.plots.waterfall(shap_val[0])
    ################################################################################
    ################################################################################
    # CatBoost Classifier
    catboost = CatBoostClassifier(verbose=False)
    catboost.fit(X_train, Y_train)
    acc = print_evaluation_metrics('catboost' ,catboost.predict(X_test), Y_test)
    result.extend(acc)
    # shap.plots.waterfall(catboos_val[0])
    ################################################################################
    '''
    if file_nm in shap_cal:
        if stock_nm in check_shap.name.unique(): continue
    
        logistic_explainer = shap.Explainer(logistic, X_test)
        logistic_val = logistic_explainer(X_test)
        tmp = pd.DataFrame(logistic_val.values)
        shap_list.append(stock_nm)
        shap_list.append('logistic_regression')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [logistic_val.base_values.mean()] + [logistic_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        svm_linear_explainer = shap.Explainer(svm_linear, X_test)
        svm_linear_val = svm_linear_explainer(X_test)
        tmp = pd.DataFrame(svm_linear_val.values)
        shap_list.append(stock_nm)
        shap_list.append('svm_linear')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [svm_linear_val.base_values.mean()] + [svm_linear_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        svc_rbf_explainer = shap.Explainer(svc_rbf.predict, X_test)
        svc_rbf_val = svc_rbf_explainer(X_test)
        tmp = pd.DataFrame(svc_rbf_val.values)
        shap_list.append(stock_nm)
        shap_list.append('svc_rbf')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [svc_rbf_val.base_values.mean()] + [svc_rbf_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        knn_explainer = shap.Explainer(knn.predict, X_test)
        knn_val = knn_explainer(X_test)
        tmp = pd.DataFrame(knn_val.values)
        shap_list.append(stock_nm)
        shap_list.append('knn')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [knn_val.base_values.mean()] + [knn_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        dtree_explainer = shap.Explainer(dtree.predict, X_test)
        dtree_val = dtree_explainer(X_test)
        tmp = pd.DataFrame(dtree_val.values)
        shap_list.append(stock_nm)
        shap_list.append('dtree')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [dtree_val.base_values.mean()] + [dtree_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        rforest_explainer = shap.Explainer(rforest.predict, X_test)
        rforest_val = rforest_explainer(X_test)
        tmp = pd.DataFrame(rforest_val.values)
        shap_list.append(stock_nm)
        shap_list.append('rforest')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [rforest_val.base_values.mean()] + [rforest_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        nn_explainer = shap.Explainer(nn.predict, X_test)
        nn_val = nn_explainer(X_test)
        tmp = pd.DataFrame(nn_val.values)
        shap_list.append(stock_nm)
        shap_list.append('NN')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [nn_val.base_values.mean()] + [nn_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        xgboost_explainer = shap.Explainer(xgboost)
        xgboost_val = xgboost_explainer(X_test)
        tmp = pd.DataFrame(xgboost_val.values)
        shap_list.append(stock_nm)
        shap_list.append('xgboost')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [xgboost_val.base_values.mean()] + [xgboost_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
        
        catboost_explainer = shap.Explainer(catboost)
        catboos_val = catboost_explainer(X_test)
        tmp = pd.DataFrame(catboos_val.values)
        shap_list.append(stock_nm)
        shap_list.append('catboost')
        shap_list.extend(tmp.mean().tolist() + tmp.std().tolist() + [catboos_val.base_values.mean()] + [catboos_val.base_values.std()])
        shap_lists.append(shap_list)
        shap_list = list()
    '''    
        
    results.append(result)
    if i % 100 == 0:
        results = pd.DataFrame(results, columns = result_cols)
        results.to_csv("{}.csv".format(save_file_path), mode='a', header=False, index = False)
        results = list()
        
        # shap_lists = pd.DataFrame(shap_lists, columns = shap_cols)
        # shap_lists.to_csv("{}.csv".format(save_file_shap_path), mode='a', header=False, index = False)
        # shap_lists = list()
        
results = pd.DataFrame(results, columns = result_cols)
results.to_csv("{}.csv".format(save_file_path), mode='a', header=False, index = False)

# shap_lists = pd.DataFrame(shap_lists, columns = shap_cols)
# shap_lists.to_csv("{}.csv".format(save_file_shap_path), mode='a', header=False, index = False)
