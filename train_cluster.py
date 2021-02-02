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
from tslearn.shapelets import LearningShapelets
from tslearn.shapelets import grabocka_params_to_shapelet_size_dict as shapelet_size_dict
from tslearn.preprocessing import TimeSeriesScalerMeanVariance as Scaler
from tslearn.preprocessing import TimeSeriesResampler as Resampler

import tensorflow as tf


def read_data(file_data, sz):
    f = open(file_data, "r")
    lines = list(csv.reader(f))
    lines.pop(0)
    f.close()
    col_value = []
    col_file_id = []
    for row in lines:
        col_value.append(float(row[2]))
        col_file_id.append(int(float(row[-1])))
    cols = {"value": col_value, "file_id": col_file_id}
    df = pd.DataFrame(cols, columns=["value", "file_id"])

    data = {}
    for key, group in df.groupby(["file_id"]):
        value = group.values[:, 0]
        value = list(value[::-1])
        data[key] = value
    for key in data.keys():
        X = Resampler(sz).fit_transform(data[key])
        X = Scaler().fit_transform(X)
        '''
        X = np.squeeze(X, axis=0)
        X = Scaler().fit_transform(X)
        X = np.expand_dims(X, axis=0)
        '''
        data[key] = X
    data_temp = list(data.items())
    data = dict(sorted(data_temp, key=lambda x: x[0]))
    
    return data


def read_labels(file_labels):
    df_labels = pd.read_excel(file_labels)
    df_labels["file_id"] = df_labels["file"] * 1000 + df_labels["id"]
    df_labels = df_labels.sort_values(by="file_id", ascending=True)
    labels = df_labels["lable"].to_numpy()
    labels = np.where(labels > 10, labels, labels * 10)
    labels = (labels / 10).astype(np.int16)
    
    return labels


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
        plot_clusters(euclidean_pred, data, n_clusters, "Euclidean K-means")
    elif method == "dtw":
        dtw_pred = KMeans(n_clusters=n_clusters, n_init=2, metric="dtw", verbose=True, 
                            max_iter_barycenter=50, random_state=seed)
        plot_clusters(dtw_pred, data, n_clusters, "DTW K-means")
    elif method == "softdtw":
        softdtw_pred = KMeans(n_clusters=n_clusters, n_init=2, metric="softdtw", verbose=True, 
                                metric_params={"gamma": 0.01}, max_iter_barycenter=50, random_state=seed)
        plot_clusters(softdtw_pred, data, n_clusters, "soft-DTW K-means")
    elif method == "kernel":
        kernel_pred = KernelKMeans(n_clusters=n_clusters, n_init=50, kernel="gak", verbose=True,
                                    kernel_params={"sigma": "auto"}, random_state=seed)
        plot_clusters(kernel_pred, data, n_clusters, "kernel K-means")
    elif method == "kshape":
        kshape_pred = KShape(n_clusters=n_clusters, n_init=2, verbose=True, random_state=seed)
        plot_clusters(kshape_pred, data, n_clusters, "K-Shape")
    else:
        print("required method not included!")


def plot_clusters(pred, data, n_clusters, method):
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
            plt.plot(curve.ravel(), "silver", alpha=0.2)
        if method != "kernel K-means":
            plt.plot(pred.cluster_centers_[i].ravel(), "maroon")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.axis("off")
        plt.text(0.6, 0.85, "cluster %d" %(i+1), family="fantasy", color="navy", 
                transform=plt.gca().transAxes)
    plt.subplots_adjust(left=0.05, top=0.9, right=0.95, bottom=0.05, wspace=0.15, hspace=0.15)
    plt.suptitle(method)
    plt.savefig("%s.png" %method, format="png")
    plt.cla()
    plt.clf()
    with open("%s.txt" %method, "w") as f:
        for i in range(n_clusters):
            f.write("cluster %d :\n" %(i+1))
            for idx in data_index[y_pred == i]:
                f.write("%d " %idx)
            f.write("\n")


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
    shapelet_sizes = shapelet_size_dict(n_ts=n_ts, ts_sz=sz, n_classes=n_classes, l=0.1, r=2)
    shapelet_clf = LearningShapelets(n_shapelets_per_size=shapelet_sizes,
                                    optimizer=tf.optimizers.Adam(0.001),
                                    batch_size=32,
                                    max_iter=1000,
                                    random_state=seed,
                                    verbose=0)
    shapelet_clf.fit(X_train, y_train)
    pred = shapelet_clf.predict(X_test)
    cm = confusion_matrix(y_test, pred, labels=[(i+1) for i in range(n_classes)])

    with open("shapelets_report.txt", "w") as f:
        f.write(classification_report(y_test, pred, digits=6))
    with open("shapelets_pred.txt", "w") as f:
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
    plt.savefig("shapelets CM.png", format="png")
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
    plt.savefig("shapelets.png", format="png")
    plt.cla()
    plt.clf()

    plt.figure()
    plt.plot(np.arange(1, shapelet_clf.n_iter_+1), shapelet_clf.history_["loss"], color="navy")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.savefig("loss.png", format="png")
    plt.cla()
    plt.clf()


def main():
    # args
    seed = 0
    n_clusters = 9
    size = 400
    file_data = "data.csv"
    file_labels = "labels.xlsx"
    dir_results = "clustering_all_file"

    data = read_data(file_data, sz=size)
    labels = read_labels(file_labels)
    # choose method from "ed", "dtw", "softdtw", "kernel", "kshape"
    method = "kshape"
    '''
    clustering(data, method, n_clusters, seed)
    evaluate(dir_results, labels)
    '''
    classify(data, labels, seed)


if __name__ == "__main__":
    main()
