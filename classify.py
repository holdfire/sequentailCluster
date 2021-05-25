"""
Using tslearn to cluster sequential data.
@author: 
@date: 2021-02-01 
"""

import os
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import sklearn
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from tslearn.utils import ts_size
from tslearn.clustering import TimeSeriesKMeans as KMeans
from tslearn.clustering import KernelKMeans
from tslearn.clustering import KShape
from tslearn.preprocessing import TimeSeriesScalerMeanVariance as Scaler
from tslearn.preprocessing import TimeSeriesResampler as Resampler
from tslearn.shapelets import LearningShapelets
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict as shapelet_size_dict




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
        
        

# 以下是分类用到的函数
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
    X_train, X_test, y_train, y_test = train_test_split(data_value, labels, test_size=0.4, 
                                                        random_state=seed, stratify=labels)
    shapelet_sizes = shapelet_size_dict(n_ts=n_ts, ts_sz=sz, n_classes=n_classes, l=0.02, r=2)
    shapelet_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                    optimizer=tf.optimizers.Adam(0.001),
                                    batch_size=32,
                                    max_iter=5000,
                                    random_state=seed,
                                    verbose=0)
    shapelet_clf.fit(X_train, y_train)
    pred = shapelet_clf.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[(i+1) for i in range(n_classes)])

    with open("./classify_result/shapelets_report.txt", "w") as f:
        f.write(classification_report(y_test, pred, digits=6))
    with open("./classify_result/shapelets_pred.txt", "w") as f:
        all_pred = shapelet_clf.predict(data_value)
        f.write("file_id, true label, pred label\n")
        for out1, out2, out3 in zip(data_index, labels, all_pred):
            f.write("%d, %d, %d\n" %(out1, out2, out3))
    
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment="center", verticalalignment="center")
    plt.xticks([i for i in range(n_classes)], [(i+1) for i in range(n_classes)])
    plt.yticks([i for i in range(n_classes)], [(i+1) for i in range(n_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Shapelets Confusion Matrix", y=1.1)
    plt.savefig("./classify_result/shapelets CM.png", format="png")
    plt.cla()
    plt.clf()
    
    plt.figure()
    for i, sz in enumerate(shapelet_sizes.keys()):
        plt.subplot(len(shapelet_sizes), 1, i + 1)
        plt.title("%d shapelets of size %d" %(shapelet_sizes[sz], sz))
        for shape in shapelet_clf.shapelets_:
            if ts_size(shape) == sz:
                plt.plot(shape.ravel())
                plt.xlim(0, sz-1)
    plt.tight_layout()
    plt.savefig("./classify_result/shapelets.png", format="png")
    plt.cla()
    plt.clf()

    plt.figure()
    plt.plot(np.arange(1, shapelet_clf.n_iter_+1), shapelet_clf.history_["loss"], color="navy")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.savefig("./classify_result/loss.png", format="png")
    plt.cla()
    plt.clf()




def main():

    # args
    seed = 0
    n_clusters = 5

    size = 30
    src_file = "./data/table_v2.csv"
    data = read_data_2(src_file, sz=size)

    file_labels = "./data/labels.csv"
    labels = read_labels_2(file_labels)
    # print(len(data))

    # 分类的函数
    # classify(data, labels, seed)
    


if __name__ == "__main__":
    main()
