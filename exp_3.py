# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 02:50:51 2022

@author: sapzzil
"""
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
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

def print_accuracy(name, pred, Y_test):
    print("{} model Accuracy = {}%".format(name, 100*np.sum(pred == Y_test)/len(Y_test)))
    return 100*np.sum(pred == Y_test)/len(Y_test)

# data set 구성
# n일 단위로 나눠서 2일 후의 상승/하락을 맞춘다
n = 5
predict_day = 2
data_dir = 'US_Stock_market'
result_cols = ['name', 'total_len','train_len','test_len',
               'logistic-regression', 'svm_linear', 'svc_rbf', 'knn', 'dtree',
               'rforest', 'NN', 'xgboost', 'catboost']
results = list()
skipped = list()
shap_lists = list()
file_list = [f for f in os.listdir(data_dir) if f.split('_')[-1] != 'intraday.csv']
shap_cal = random.sample(file_list, int(len(file_list) * 0.05))
check = pd.read_csv('result_3.csv')
for i, file_nm in enumerate(file_list):
    if file_nm.split('.')[0] in check.name.unique(): continue
    data_path = os.path.join(data_dir, file_nm)
    data = pd.read_csv(data_path)
    data = data.sort_values('date')
    data.index = data.date
    data = data[['4. close']]
    diff_data = data.diff(predict_day).shift(-predict_day)[:-2]
    Y = [1 if x >= 0 else 0 for x in diff_data.values]
    data = data[2:]
    data['Y'] = Y
    dataset = [[data.to_numpy()[i:i+n,0], data.to_numpy()[i+n-1,1]] for i in range(len(data) - n)]
    dataset = np.array(dataset)
    dataset = [x.tolist()+[y] for x,y in dataset]
    dataset = pd.DataFrame(dataset, columns=['n-4','n-3','n-2','n-1','n-0','Y'])
    X = dataset.iloc[:,:-1]
    ## normalize
    X = (X - X.mean()) / X.std()
    Y = dataset.iloc[:,-1]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2)
    if len(Y_train.unique()) == 1:
        skipped.append(file_nm)
        continue
    shap_list = list()
    result = list()
    # stock name
    result.append(file_nm.split('.')[0])
    # total_len
    result.append(len(X))
    # train_len
    result.append(len(X_train))
    # test_len
    result.append(len(X_test))
    ################################################################################
    # logistic regression
    logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    acc = print_accuracy('logistic-regression' ,logistic.predict(X_test), Y_test)
    result.append(acc)
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
    acc = print_accuracy('svm_linear' ,svm_linear.predict(X_test), Y_test)
    result.append(acc)
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
    acc = print_accuracy('svc_rbf' ,svc_rbf.predict(X_test), Y_test)
    result.append(acc)
    # shap.plots.waterfall(svc_rbf_val)
    # svc_rbf_explainer = shap.KernelExplainer(svc_rbf.predict_proba, X_train)
    # svc_rbf_val = svc_rbf_explainer.shap_values(X_test)
    # shap.force_plot(svc_rbf_explainer.expected_value[0], svc_rbf_val[0], X_test)
    ################################################################################
    ################################################################################
    # K-nearest neighbors classifier
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    acc = print_accuracy('knn' ,knn.predict(X_test), Y_test)
    result.append(acc)
    # shap.plots.waterfall(knn_val[0])
    # knn_explainer = shap.KernelExplainer(knn.predict_proba, X_train)
    # knn_val = knn_explainer.shap_values(X_test.iloc[0,:])
    # shap.force_plot(knn_explainer.expected_value[0], knn_val[0], X_test.iloc[0,:])
    ################################################################################
    ################################################################################
    # Decision Tree classifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, Y_train)
    acc = print_accuracy('dtree' ,dtree.predict(X_test), Y_test)
    result.append(acc)
    # shap.plots.waterfall(dtree_val[0])
    # shap.force_plot(dtree_explainer.expected_value[0], dtree_val[0], X_test)
    ################################################################################
    ################################################################################
    # Random Forest classifier
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    rforest.fit(X_train, Y_train)
    acc = print_accuracy('rforest' ,rforest.predict(X_test), Y_test)
    result.append(acc)
    # shap.force_plot(rforest_explainer.expected_value[0], rforest_val[0], X_test)
    ################################################################################
    ################################################################################
    # Multi Layer perceptron classifier
    nn = MLPClassifier(solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=0)
    nn.fit(X_train, Y_train)
    acc = print_accuracy('nn' ,nn.predict(X_test), Y_test)
    result.append(acc)
    # shap.force_plot(nn_explainer.expected_value[0], nn_val[0], X_test)
    ################################################################################
    ################################################################################
    # XGBoost Classifier
    xgboost = XGBClassifier()
    xgboost.fit(X_train, Y_train)
    acc = print_accuracy('xgboost' ,xgboost.predict(X_test), Y_test)
    result.append(acc)
    # shap.plots.waterfall(shap_val[0])
    ################################################################################
    ################################################################################
    # CatBoost Classifier
    catboost = CatBoostClassifier(verbose=False)
    catboost.fit(X_train, Y_train)
    acc = print_accuracy('catboost' ,catboost.predict(X_test), Y_test)
    result.append(acc)
    # shap.plots.waterfall(catboos_val[0])
    ################################################################################
    if file_nm in shap_cal:
        logistic_explainer = shap.Explainer(logistic, X_test)
        logistic_val = logistic_explainer(X_test)
        shap_list.append(logistic_val)
        svm_linear_explainer = shap.Explainer(svm_linear, X_test)
        svm_linear_val = svm_linear_explainer(X_test)
        shap_list.append(svm_linear_val)
        svc_rbf_explainer = shap.Explainer(svc_rbf.predict, X_test)
        svc_rbf_val = svc_rbf_explainer(X_test)
        shap_list.append(svc_rbf_val)
        knn_explainer = shap.Explainer(knn.predict, X_test)
        knn_val = knn_explainer(X_test)
        shap_list.append(knn_val)
        dtree_explainer = shap.Explainer(dtree.predict, X_test)
        dtree_val = dtree_explainer(X_test)
        shap_list.append(dtree_val)
        rforest_explainer = shap.Explainer(rforest.predict, X_test)
        rforest_val = rforest_explainer(X_test)
        shap_list.append(rforest_val)
        nn_explainer = shap.Explainer(nn.predict, X_test)
        nn_val = nn_explainer(X_test)
        shap_list.append(nn_val)
        xgboost_explainer = shap.Explainer(xgboost)
        xgboost_val = xgboost_explainer(X_test)
        shap_list.append(xgboost_val)
        catboost_explainer = shap.Explainer(catboost)
        catboos_val = catboost_explainer(X_test)
        shap_list.append(catboos_val)
        shap_lists.append(shap_list)
    results.append(result)
    if i % 100 == 0:
        results = pd.DataFrame(results, columns = result_cols)
        results.to_csv("{}.csv".format('result_3'), mode='a', header=False, index = False)
        results = list()
results = pd.DataFrame(results, columns = result_cols)
results.to_csv("{}.csv".format('result_3'), mode='a', header=False, index = False)