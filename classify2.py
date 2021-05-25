import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsfresh
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.cluster import KMeans
import sklearn
from sklearn import preprocessing
# import torch
import random
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,RFE,SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from math import modf
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics


def read_data_2(src_file, sz=30):
    with open(src_file, 'r', encoding='gbk') as fr:
        lines = list(fr.readlines())
        lines.pop(0)
    data = {}
    for line in lines:
        parts = line.strip().split(",")
        key = str2int(parts[0])
        if key is None:
            continue
        data[key] = np.array([[[float(x)] for x in parts[1:]]])

    # 做最大值-最小值归一化，max=28.82， min=8.06
    dele = []
    for key, value in data.items():
        data[key] = (data[key] - 8.06) / (28.82 - 8.06)
        if np.any(data[key] < 0) or np.any(data[key] > 1):
            dele.append(key)
            # print(dele)
    # for i in dele:
        # print(i, data[i])
        # data.pop(i)

    # print(data[16007])
    return data


def str2int(s):
    """
    把仓号名称改为整数
    """
    if "T" in s:
        return None
    if not "#-" in s:
        return None
    parts = s.strip().split("#-")
    if len(parts[1]) > 3:
        return None

    if int(parts[0])  < 10:
        part1 = "0" + parts[0]
        return int(part1 + parts[1])
    else:
        return int(parts[0] + parts[1])
        

def score_result(y_true,y_predict):
    acc = metrics.accuracy_score(y_true,y_predict)
    print(acc)
    recall = metrics.recall_score(y_true,y_predict,average='weighted')
    print(recall)
    f1 = metrics.f1_score(y_true,y_predict,average='weighted')
    print(f1)
    prec = metrics.precision_score(y_true,y_predict,average='weighted')
    print(prec)
    cm = metrics.confusion_matrix(y_true,y_predict)
    return acc,recall,f1,prec,cm


def read_labels_2(file_labels):
    with open(file_labels, 'r', encoding='gbk') as fr:
        lines = fr.readlines()
        res = []
        for line in lines:
            res.append(int(line.strip().split(",")[1]))
        labels = np.array(res)
        print("labesl after sort, count: ", labels.shape)
    return labels


def classify(data, labels, seed):
    n_ts = len(data)
    sz = random.choice(list(data.values())).shape[1]
    n_classes = len(set(labels))
    data_index = np.zeros((n_ts, 1))
    data_value = np.zeros((n_ts, sz, 1))
    for i, (key, value) in enumerate(data.items()):
        data_index[i, :] = key
        data_value[i, :, :] = value
    data_value = np.reshape(data_value, data_value.shape[:2])
    x_train,x_test,y_train,y_test = train_test_split(data_value, labels, test_size=0.2, 
                                                        random_state=seed, stratify=labels)

    print(x_train.shape)                                                        
    print(y_train.shape)   


    
    # 随机森林
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("Random Forest: ")
    acc,recall,f1,prec,cm  = score_result(y_test,y_pred)
    # print(acc,recall,f1,prec,cm)

    n_classes = 6
    plt.figure(num=1,figsize=(15,15),dpi=200)
    plt.matshow(cm,cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')
    plt.xticks([i for i in range(n_classes)],[(i+1) for i in range(n_classes)])
    plt.yticks([i for i in range(n_classes)],[(i+1) for i in range(n_classes)])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Random Forest',y=1.1)
    plt.savefig('./classify_result2/RF.png',format = 'png')



    # GradientBoostingClassifier
    model = GradientBoostingClassifier()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("GradientBoostingClassifier: ")
    acc,recall,f1,prec,cm  = score_result(y_test,y_pred)
    plt.matshow(cm,cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('GradientBoosting')
    plt.savefig('./classify_result2/GradientBoosting.png',format = 'png')



    # AdaBoostClassifier
    model = AdaBoostClassifier(learning_rate = 0.1,n_estimators=60)
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    print("AdaBoostClassifier: ")
    acc,recall,f1,prec,cm  = score_result(y_test,y_pred)
    plt.matshow(cm,cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('AdaBoost')
    plt.savefig('./classify_result2/AdaBoost.png',format = 'png')



    # XGB
    model = xgb.XGBClassifier(learning_rate=0.1,
                       n_estimators=1000,         # 树的个数--1000棵树建立xgboost
                      max_depth=6,               # 树的深度
                       min_child_weight = 1,      # 叶子节点最小权重
                      gamma=0.,                  # 惩罚项中叶子结点个数前的参数
                      subsample=0.8,             # 随机选择80%样本建立决策树
                       colsample_btree=0.8,       # 随机选择80%特征建立决策树
                       objective='multi:softmax', # 指定损失函数
                       scale_pos_weight=1,        # 解决样本个数不平衡的问题
                       random_state=27            # 随机数
                       )
    model.fit(x_train,
            y_train,
            eval_set = [(x_test,y_test)],
            eval_metric = "mlogloss",
            early_stopping_rounds = 10,
            verbose = True)
    y_pred = model.predict(x_test)
    print("XGB: ")
    acc,recall,f1,prec,cm  = score_result(y_test,y_pred)
    plt.matshow(cm,cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x,y],xy=(x,y),horizontalalignment='center',verticalalignment='center')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('xgboost')
    plt.savefig('./classify_result2/xgboost.png',format = 'png')





def main():
    # args
    dir_results = "clustering_all_file"
    src_file = "./data/table_v2.csv"
    seed = 0
    n_clusters = 5
    size = 30
    data = read_data_2(src_file, sz=size)

    # 分类的函数
    file_labels = "./data/labels.csv"
    labels = read_labels_2(file_labels)
    classify(data, labels, seed)
    




if __name__ == "__main__":
    main()
