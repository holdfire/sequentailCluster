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
        

def read_labels_2(file_labels):
    with open(file_labels, 'r', encoding='gbk') as fr:
        lines = fr.readlines()
        res = []
        for line in lines:
            res.append(int(line.strip().split(",")[1]))
        labels = np.array(res)
        print("labesl after sort, count: ", labels.shape)
    return labels


def get_file_name(dir_results):
    file_name = []
    file_path = []
    for root, dirs, files in os.walk(dir_results):
        for f in files:
            if os.path.splitext(f)[1] == ".txt":
                file_name.append(os.path.splitext(f)[0])
                file_path.append(os.path.join(root, f))
    return file_name, file_path


def evaluate(dir_results, labels):
    preds_name, preds_path = get_file_name(dir_results)
    preds = []
    for name, path in zip(preds_name, preds_path):
        with open(path, "r") as f:
            print(path)
            pred_lines = f.readlines()
            pred = []
            file_id = []
            for k in range(len(pred_lines)):
                if "cluster" in pred_lines[k]:
                    temp_label = int(pred_lines[k].split()[1])
                    temp_index = list(map(int, pred_lines[k+1].split()))
                    pred.extend([temp_label for i in range(len(temp_index))])
                    file_id.extend(temp_index)
            file_id, pred = zip(*sorted(zip(file_id, pred)))
            preds.append(np.array(pred))
    
    with open("clustering_eval.txt", "w") as f:
        for name, pred in zip(preds_name, preds):
            ari = ARI(labels, pred)
            nmi = NMI(labels, pred)
            f.write(name + "\n")
            f.write("ARI: %f\n" %ari)
            f.write("NMI: %f\n" %nmi)




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


    # 聚类评价
    dir_results = "./cluster_result"
    evaluate(dir_results, labels)



if __name__ == "__main__":
    main()
