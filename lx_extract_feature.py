import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import tsfresh
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.cluster import KMeans
from sklearn.cluster import spectral_clustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import sklearn
from sklearn import metrics
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,RFE,SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score



def stack_table(src_file, dst_file):
    """
    把原来的table_v2.csv重新拼接一下
    """
    with open(src_file, 'r') as fr, open(dst_file, 'w') as fw:
        fw.writelines("id,date,value\n")

        df = pd.read_csv(src_file, index_col=[0], encoding='gbk')
        for i in df.index:
            for j in df.columns:
                content = "{},{},{}\n".format(i, j, str(df.loc[i, j]))
                # print(content)
                # exit(0)
                fw.writelines(content)
    return dst_file



def extract(src_file, feature_save_dir, lable_file):
    """
    从src_file中提取特征，结果保存到feature_original.csv
    """
    df = pd.read_csv(src_file)
    data = df[['date', 'value', 'id']]

    # 提特征并保存
    # settings = ComprehensiveFCParameters()
    # data_feature = tsfresh.extract_features(data, default_fc_parameters=settings, column_id="id", column_sort="date")
    # data_feature.to_csv(os.path.join(feature_save_dir, "feature_original.csv"), encoding='gbk')
    # 只提一次，下次可以直接从文件中加载
    data_feature = pd.read_csv(os.path.join(feature_save_dir, "feature_original.csv"), index_col=[0])
    print(data_feature.shape)

    # 处理label文档
    labels_df = pd.read_csv(lable_file)
    labels_df['file_id'] = labels_df.iloc[:,0]
    labels_df['label'] = labels_df.iloc[:,1]
    labels_df = labels_df.sort_values(by="file_id", ascending=True)

    # 剔除0与na
    data_feature.dropna(axis=1, how='any', inplace=True)
    data_feature.dropna(axis=0, how='any', inplace=True)
    print(data_feature.shape)
    
    # 删除包含inf的列
    data_feature = data_feature.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    print(data_feature.shape)

    # 归一化
    data_feature.to_csv(os.path.join(feature_save_dir, "feature_delete_nanInf.csv"), encoding='gbk')
    print(data_feature)
    data_scale = preprocessing.scale(data_feature, axis=0)
    # data = torch.from_numpy(data)
    # print(data)
    # thresholds = [0.1,0.3,0.5,0.7,0.9,1.0]
    # for t in thresholds:
    ns = [5, 10, 20, 30, 50, 100]
    for n in ns:
        # 特征选择
        var = VarianceThreshold(threshold = 0.1)
        data_feature_selected = pd.DataFrame(var.fit_transform(data_scale))

        # 剔除0与na
        data_feature_selected.dropna(axis=1, how='any', inplace=True)

        # 使用PCA降维
        pca = PCA(n_components= n)
        data20 = pca.fit_transform(data_feature_selected)

        # 保存PCA降维后的结果
        res = pd.DataFrame(data20, index=data_feature.index)
        res.to_csv(os.path.join(feature_save_dir, "feature_PCA_{}.csv".format(str(n))), encoding='gbk')
        


    #     label_kmeans = KMeans(n_clusters = 9).fit_predict(data20)
    #     labels = labels_df['label'].to_numpy()
    #     labels = np.where(labels > 10, labels, labels * 10)
    #     labels = (labels/10).astype(np.int16)
    #     linkages = ['ward', 'average', 'complete','single']
    #     ac = AgglomerativeClustering(linkage=linkages[0], n_clusters=9)
    #     ac.fit(data20)
    #     label = ac.labels_
    #     nmi = normalized_mutual_info_score(label,labels)
    #     ari = adjusted_rand_score(label, labels)
    #     nmi_k = normalized_mutual_info_score(label_kmeans,labels)
    #     ari_k = adjusted_rand_score(label_kmeans, labels)
    #     # print(nmi_k, ari_k)
    #     print(nmi,ari)

    # silhouettescore = []
    # for i in range(2, 30):
    #     kmeans = KMeans(n_clusters=i, random_state=123).fit(data20)
    #     score = silhouette_score(data20, kmeans.labels_)
    #     silhouettescore.append(score)
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(2, 30), silhouettescore, linewidth=1.5, linestyle='-')
    # plt.show()
    # model = TSNE()
    # Y = model.fit_transform(data20)
    #
    # plt.scatter(Y[:, 0], Y[:, 1], 20, label)
    # plt.savefig("tsne.png")  # 保存图片
    # plt.show()
    # means = []
    # ID = []
    # for i in range(1, 7):
    #     df_file = df[df['file'] == i]
    #     ids = df_file['id'].unique()
    #     for id in ids:
    #         df_id = df_file[df_file['id'] == id]
    #         value = df_id['value']
    #         # df_id.loc['value'] = (df_id['value']-df_id['value'].min())/(df_id['value'].max() - df_id['value'].min())
    #         DF = df.append(value, ignore_index=True)
    # DF.to_csv('normalized.csv')
    

    





if __name__ == "__main__":

    src_file = "./data/table_v2.csv"
    dst_file = "./data/table_v4.csv"
    stack_table(src_file, dst_file)


    src_file = "./data/table_v4.csv"
    dst_file = "./data/"
    lable_file = "./data/labels.csv"
    extract(src_file, dst_file, lable_file)
