### 时间序列的聚类分析
（1）使用pip3安装所需的python包：
linux下使用shell命令：`sh environ.sh`;  
windows中可使用命令：`pip3 install -c requirements.txt`；  
  
（2）原始数据存放于data目录下，首先使用process.py处理原始文件:  
`python3  process.py`

（3）基于特征的聚类：
+ 第一步提取最高温度，以及对应的收仓日期；
+ 第二步，直接画图或聚类；

（4）基于时间序列的聚类：
+ 第一步：选取收仓日期后的30天长度的序列做聚类；

（5）基于最高温度，将序列划分为两段，并保存；

