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
    # for i in dele:
    #     # print(i, data[i])
    #     data.pop(i)

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
        


def factorize(num):
    factor = round(np.sqrt(num))
    while factor <= num :
        if num % factor == 0:
            break
        factor += 1
    if (factor == num) or (factor / (num/factor) > 3):
        return factorize(num+1)
    return (int(num/factor), int(factor))


def clustering(data, method, n_clusters, seed):
    if method == "ed":
        euclidean_pred = KMeans(n_clusters=n_clusters, verbose=True, random_state=seed)
        plot_clusters_separate(euclidean_pred, data, n_clusters, "Euclidean K-means")
    elif method == "dtw":
        dtw_pred = KMeans(n_clusters=n_clusters, n_init=2, metric="dtw", verbose=True, 
                            max_iter_barycenter=50, random_state=seed)
        plot_clusters(dtw_pred, data, n_clusters, "DTW K-means")
    elif method == "softdtw":
        softdtw_pred = KMeans(n_clusters=n_clusters, n_init=2, metric="softdtw", verbose=True, 
                                metric_params={"gamma": 0.01}, max_iter_barycenter=50, random_state=seed)
        plot_clusters(softdtw_pred, data, n_clusters, "soft-DTW K-means")
    elif method == "kernel":
        kernel_pred = KernelKMeans(n_clusters=n_clusters, n_init=5, kernel="gak", verbose=True,
                                    kernel_params={"sigma": "auto"}, random_state=seed)
        plot_clusters(kernel_pred, data, n_clusters, "kernel K-means")
    elif method == "kshape":
        kshape_pred = KShape(n_clusters=n_clusters, n_init=6, verbose=True, random_state=seed)
        plot_clusters(kshape_pred, data, n_clusters, "K-Shape")
    else:
        print("required method not included!")


def plot_clusters(pred, data, n_clusters, method, cluster_result_dir="./cluster_result", norm=True):
    # 保存图片的目录
    if not os.path.exists(cluster_result_dir):
        os.mkdir(cluster_result_dir)

    n_ts = len(data)
    sz = random.choice(list(data.values())).shape[1]
    data_index = np.zeros((n_ts, 1))
    data_value = np.zeros((n_ts, sz, 1))
    for i, (key, value) in enumerate(data.items()):
        data_index[i, :] = key
        data_value[i, :, :] = value
    y_pred = pred.fit_predict(data_value)
    plt.figure()
    row, col = factorize(n_clusters)
    for i in range(n_clusters):
        plt.subplot(row, col, 1+i)
        for curve in data_value[y_pred == i]:
            if norm:
                new_value = curve.ravel() * (28.82 - 8.06) + 8.06
            plt.plot(new_value, "silver", alpha=0.2)
        if method != "kernel K-means":
            if norm:
                new_center = pred.cluster_centers_[i].ravel() * (28.82 - 8.06) + 8.06
            plt.plot(new_center, "maroon")
        plt.xlim(0, sz)
        plt.ylim(8, 30)
        # plt.axis("off")
        plt.text(0.6, 0.85, "cluster %d" %(i+1), family="fantasy", color="navy", 
                transform=plt.gca().transAxes)
    plt.subplots_adjust(left=0.05, top=0.9, right=0.95, bottom=0.05, wspace=0.15, hspace=0.15)
    plt.suptitle(method)
    plt.savefig(os.path.join(cluster_result_dir, "%s.png" %method), format="png")
    plt.cla()
    plt.clf()
    with open(os.path.join(cluster_result_dir, "%s.txt" %method), "w") as f:
        for i in range(n_clusters):
            f.write("cluster %d :\n" %(i+1))
            for idx in data_index[y_pred == i]:
                f.write("%d " %idx)
            f.write("\n")



def plot_clusters_separate(pred, data, n_clusters, method, cluster_result_dir="./cluster_result", norm=True):
    # 保存图片的目录
    if not os.path.exists(cluster_result_dir):
        os.mkdir(cluster_result_dir)

    n_ts = len(data)
    sz = random.choice(list(data.values())).shape[1]
    data_index = np.zeros((n_ts, 1))
    data_value = np.zeros((n_ts, sz, 1))
    for i, (key, value) in enumerate(data.items()):
        data_index[i, :] = key
        data_value[i, :, :] = value
    y_pred = pred.fit_predict(data_value)
    row, col = factorize(n_clusters)
    for i in range(n_clusters):
        for curve in data_value[y_pred == i]:
            if norm:
                new_value = curve.ravel() * (28.82 - 8.06) + 8.06
            plt.plot(new_value, "silver", alpha=0.2)
        if method != "kernel K-means":
            if norm:
                new_center = pred.cluster_centers_[i].ravel() * (28.82 - 8.06) + 8.06
            plt.plot(new_center, "maroon")
        plt.xlim(0, sz)
        plt.ylim(8, 30)
        # plt.axis("off")
        plt.text(0.6, 0.85, "cluster %d" %(i+1), family="fantasy", color="navy", 
                transform=plt.gca().transAxes)
        plt.savefig(os.path.join(cluster_result_dir, "{method}_{cluster}.png".format(method=method, cluster=str(i))), format="png")
        plt.cla()
        plt.clf()
   
    with open(os.path.join(cluster_result_dir, "%s.txt" %method), "w") as f:
        for i in range(n_clusters):
            f.write("cluster %d :\n" %(i+1))
            for idx in data_index[y_pred == i]:
                f.write("%d " %idx)
            f.write("\n")



def plot_all(src_file, save_dir = "./plot_all/"):
    """
    把table_v2.csv的每一行画出曲线
    """
    # 保存图片的目录
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

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
    
    n_ts = len(data)
    count = 0

    for key, value in data.items():
        # 把仓号命名从数字转换为原来的格式
        if key < 10000:
            name = "0" + str(key)
        else:
            name = str(key)
        name = name[:2] + "#-" + name[2:]

        x = [i+1 for i in range(value.shape[1])]
        y = [i for i in value[0, :, 0]]
        plt.plot(x, y)
        plt.title(name)
        plt.xlabel("Day")
        plt.ylabel("T")
        plt.xlim(0, len(x))
        plt.ylim(8, 30)
        plt.savefig(os.path.join(save_dir, "%s.png" %name), format="png")
        count += 1
        print("saving: ", count)
        plt.cla()
        plt.clf()
    return      
    



def main():
    # args
    seed = 0
    n_clusters = 9
    size = 400
    file_data = "./data/data.csv"
    file_labels = "labels.xlsx"
    dir_results = "clustering_all_file"
    # data = read_data(file_data, sz=size)

    # method = "kshape"
    # clustering(data, method, n_clusters, seed)
    # labels = read_labels(file_labels)
    # choose method from "ed", "dtw", "softdtw", "kernel", "kshape"
    # evaluate(dir_results, labels)


    src_file1 = "./data/table_v2.csv"
    seed = 0
    n_clusters = 5
    size = 30
    data = read_data_2(src_file1, sz=size)

    # 把所有的曲线画出来
    # plot_all(src_file1)

    method = "ed"
    clustering(data, method, n_clusters, seed)
    


if __name__ == "__main__":
    main()
