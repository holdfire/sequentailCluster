import pandas as pd
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

if __name__ == '__main__':
    df = pd.read_csv('data.csv')
    data = df[['dt', 'value', 'file_id']]
    # settings = ComprehensiveFCParameters()
    # data_feature = tsfresh.extract_features(data, default_fc_parameters=settings, column_id="file_id", column_sort="dt")

    data_feature = pd.read_csv('feature.csv')
    labels_df = pd.read_excel('标签文档.xlsx')
    labels_df['file_id'] = labels_df['file'] * 1000 + labels_df['id']
    labels_df = labels_df.sort_values(by="file_id", ascending=True)
    # 归一化
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

        pca = PCA(n_components= n)
        data20 = pca.fit_transform(data_feature_selected)

        label_kmeans = KMeans(n_clusters = 9).fit_predict(data20)
        labels = labels_df['lable'].to_numpy()
        labels = np.where(labels > 10, labels, labels * 10)
        labels = (labels/10).astype(np.int16)
        linkages = ['ward', 'average', 'complete','single']
        ac = AgglomerativeClustering(linkage=linkages[0], n_clusters=9)
        ac.fit(data20)
        label = ac.labels_
        nmi = normalized_mutual_info_score(label,labels)
        ari = adjusted_rand_score(label, labels)
        nmi_k = normalized_mutual_info_score(label_kmeans,labels)
        ari_k = adjusted_rand_score(label_kmeans, labels)
        # print(nmi_k, ari_k)
        print(nmi,ari)

    silhouettescore = []
    for i in range(2, 30):
        kmeans = KMeans(n_clusters=i, random_state=123).fit(data20)
        score = silhouette_score(data20, kmeans.labels_)
        silhouettescore.append(score)
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, 30), silhouettescore, linewidth=1.5, linestyle='-')
    plt.show()
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
