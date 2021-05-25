#!/usr/bin/python3 
# coding: utf-8

import os
import numpy as np
import pandas as pd
import csv
import argparse


def process(args, select_rows=10766):
    """
    处理./data/table_v1.csv文件
    """
    # 读取仓编码列，收仓日期及之后的列
    cols1 = [0]
    cols2 = [i+3 for i in range(args.days)]
    cols3 = cols1 + cols2

    # windows下面由xlsx另存为csv文件，其编码方式为gb2312，而不是utf-8
    df = pd.read_csv(args.src_file, nrows=select_rows, usecols=cols3, encoding='gbk')
    # 获取文件行数，列数
    rows, cols = df.shape[:2]                                      

    # 只选取混凝土日均温度这一行，每隔5行取1行
    rows_target = [i for i in range(rows) if i % 5 == 1]          
    df2 = df.iloc[rows_target, :]   
    rows, cols = df2.shape[:2]                                      
    print("Bin numbers: {} \nDays: {} ".format(rows, cols))                       
    # print(df2)

    # 读取混凝土仓号编码，并把它作为第一列
    df_count = pd.read_csv(args.src_file, nrows=select_rows, usecols=cols1, encoding='gbk')
    rows2, cols2 = df_count.shape[:2] 
    rows_count = [i for i in range(rows2) if i % 5 == 0]
    df_count2 = df_count.iloc[rows_count, :]
    df_count2_idx = [i for i in df_count2.loc[:,"仓编码"]]

    # 将第一列“混凝土日均温度”替换为“仓编码”
    df2.loc[:, "仓编码"] = df_count2_idx
    
    # 把第30天为空的行去掉（包括整行为空的情况）,把第1天为空的也去掉了
    row_del = [x for i,x in enumerate(df2.index) if df2.iat[i,args.days] is np.nan or df2.iat[i,1] is np.nan]
    # print(row_del)
    df3 = df2.drop(row_del, axis=0)            

    # 把df2保存为一个全局变量，这样在别的函数里也能访问到args.df
    args.df = df3
    # print(df2)
    # 保存df2为一个csv文件
    args.df.to_csv(args.dst_file1, encoding='gbk')
    return



def interpolation(args):
    """
    在行方向上做插值，保存结果到args.dst_file2中
    """
    df = args.df
    # 先去掉仓编码这一列
    df2 = df.drop(["仓编码"], axis=1).astype(float)
    # print(df2)
    # 然后做插值
    df3 = df2.interpolate(method=args.interp_method, axis=1) 
    # 修改行名为仓编码
    df3.index = [i for i in df["仓编码"]]
    # print(df3)

    # 保存插值后的表格到args.dst_file2
    args.df2 = df3
    args.df2.to_csv(args.dst_file2, encoding='gbk')
    return



def feature_cluster(args):
    """
    基于特征的聚类，选取特征为：最高温度，最高温度对应的收仓日期index
    """
    df = args.df2
    # 求出最大的温度
    df_feature_max = df.max(axis=1)
    # 求出最大温度对应的索引
    df_feature_day = df.idxmax(axis=1)
    # print(df_feature_day)

    # 把上面两个保存到一个新的dataframe
    df_feature = pd.DataFrame(index=df.index, columns=["max_temp", "day"])
    df_feature["max_temp"] = df_feature_max
    df_feature["day"] = df_feature_day

    # 把中文+数字全替换为数字
    old_list = ["收仓日期"] + ["收仓日期+" + str(i+1) for i in range(args.days-1)]
    new_list = [str(i) for i in range(args.days)]
    df_feature = df_feature.replace(old_list, new_list)
    # print(df_feature)

    # 保存插值后的表格到args.dst_file2
    args.df3 = df_feature
    args.df3.to_csv(args.dst_file3, encoding='gbk')
    return




def main(args):
    # 先从原始文件中，读取所需要的行列数据
    process(args)                
    # 每个仓的时间序列长度不一样，需要做插值处理成一样长
    interpolation(args)    
    # 基于特征的聚类
    feature_cluster(args)
    return


if __name__ == "__main__":

    # 缺失数据的插值方法，其中akima得到的曲线平滑
    # 缺失值主要集中在后段，默认使用nearest最近邻插值
    interp_methods = ["linear", "quadratic", "akima", "nearest"]

    # 参数解析部分，已设置默认值
    parser = argparse.ArgumentParser(description="Process original data.")
    parser.add_argument("--src_file", type=str, default="./data/table_v0.csv", help="source file to be processed")
    parser.add_argument("--dst_file1", type=str, default="./data/table_v1.csv", help="destination file，固定天数，去掉不足30天的以及第1天缺失的")
    parser.add_argument("--dst_file2", type=str, default="./data/table_v2.csv", help="destination file，经过插值以后的数据")
    parser.add_argument("--dst_file3", type=str, default="./data/table_v3.csv", help="destination file，提出的最大的温度和龄期")
    parser.add_argument("--days", type=int, default=30, help="读取收仓日期之后的天数")
    parser.add_argument("--interp_method", type=str, default="linear", choices=interp_methods, help="缺失数据的插值方法")
    args = parser.parse_args()
    
    main(args)