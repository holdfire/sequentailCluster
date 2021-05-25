import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tsfresh
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.cluster import KMeans
import sklearn
from sklearn import preprocessing
import torch
from sklearn.feature_selection import VarianceThreshold,SelectKBest,chi2,RFE,SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from math import modf
import seaborn as sns
import os


df = pd.read_csv('data.csv')
data = df[['dt','value','file_id']]

settings = ComprehensiveFCParameters()
data_feature = tsfresh.extract_features(data,default_fc_parameters=settings, column_id="file_id", column_sort="dt")

# 归一化
data_scale = preprocessing.scale(data_feature,axis=0)
#data = torch.from_numpy(data)
#print(data)


# In[10]:


# 特征选择
var=VarianceThreshold(threshold = 1.0)
data_feature_selected=pd.DataFrame(var.fit_transform(data_scale))


# In[11]:


data_feature_selected.shape


# # PCA降维至20维度

# In[12]:


# 剔除0与na
data_feature_selected.dropna(axis=1,how='any',inplace=True)
data_feature_selected.shape


# In[13]:


pca = PCA(n_components = 20,copy=False)


# In[14]:


#data20 = pca.fit(data_feature_selected)
data20 = np.load("data20.npy")


# In[ ]:





# # 聚类

# In[67]:


label = KMeans(n_clusters=15).fit_predict(data20)


# In[68]:


print(label)


# In[45]:


model = TSNE()
Y = model.fit_transform(data20)


# # 按类别保存图片

# In[109]:


for i in np.unique(label):
    savePath = "./result_MM/label_%i" % i
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    file_id = data_feature.index[label==i]
    for file_id_ in file_id:        
        file = int(file_id_//1000)
        #file = file.astype(np.int16)
        id_ = int(file_id_ - file*1000)
        #id_ = id_.astype(np.int16)
        df_file = df[df['file']==file]
        df_id = df_file[df_file['id']==id_]
        df_id_1 = (df_id-df_id.min())/(df_id.max()-df_id.min())
        df_id_1.plot(x='dt', y='value')
        #plt.subplot(6,2,j)
        plt.title("file = %i,id = %i" % (file,id_))
        plt.savefig(os.path.join(savePath, 'file%i_id%i.png'% (file,id_)))


# In[106]:


file_id = 1073
file = int(file_id_//1000)
print(file)
id_ = int(file_id - file*1000)
print(id_)


# In[93]:


a = df['file']==1
b =pd.DataFrame(label==1)
print(a)
print(b)


# # 画图

# In[69]:


colors = ["windows blue", "amber", 
          "greyish", "faded green", 
          "dusty purple","royal blue","lilac",
          "salmon","bright turquoise",
          "dark maroon","light tan",
          "orange","orchid",
          "sandy","topaz",
          "fuchsia","yellow",
          "crimson","cream",
          "grey","grass"
          ]
current_palette = sns.xkcd_palette(colors)


# In[72]:


points = Y
plt.figure(figsize=(8,8))
for i in np.unique(label):
    plt.scatter( points[label==i,0], points[label==i,1], c=current_palette[i], label=i, s=20)    
plt.legend(scatterpoints=1,loc='upper center',
           bbox_to_anchor=(0.5,-0.08),ncol=6,
           fancybox=True,
           prop={'size':8}
          )
plt.savefig("tsne.png") #保存图片
plt.show()


# In[71]:





# In[34]:


#data_feature.columns.values.tolist()
file_id = np.array(data_feature._stat_axis.values.tolist())
type(file_id)


# In[75]:


file = file_id//1000
file = file.astype(np.int16)
id_ = file_id - file*1000
id_ = id_.astype(np.int16)
#print(file)
#print(id_)


# # label为file

# In[76]:


np.unique(file)


# In[77]:


plt.figure(figsize=(8,8))
for i in np.unique(file):
    plt.scatter( points[file==i,0], points[file==i,1], c=current_palette[i], label=i, s=20)    
plt.legend(scatterpoints=1,loc='upper center',
           bbox_to_anchor=(0.5,-0.08),ncol=6,
           fancybox=True,
           prop={'size':8}
          )
plt.savefig("tsne_file.png") #保存图片
plt.show()


# # label为id

# In[79]:


np.unique(id_)


# In[78]:


plt.figure(figsize=(8,8))
for i in np.unique(id_):
    plt.scatter( points[id_==i,0], points[id_==i,1], c=current_palette[i], label=i, s=20)    
plt.legend(scatterpoints=1,loc='upper center',
           bbox_to_anchor=(0.5,-0.08),ncol=6,
           fancybox=True,
           prop={'size':8},
           title = 'id'
          )
plt.savefig("tsne_id.png") #保存图片
plt.show()


# # 其他

# In[ ]:


df_file = df[df['file']==1]
ids = df_file['id'].unique()
df_id = df_file[df_file['id']==0]
x=df_id[['value','id']]
print(df_file)


# In[ ]:


temp


# In[ ]:


a = data_feature_selected.isna().sum().sum()
b = data_feature_selected.isin([0]).sum().sum()
print(a)
print(b)


# In[ ]:


data_feature_selected.describe()

