#!/usr/bin/python3 
# coding: utf-8

import os
import numpy as np
import pandas as pd
import scipy
import csv
import argparse


def process(args):
    """
    处理./data/table_v1.csv文件
    """
    # 读取仓编码列，收仓日期及之后的列
    cols1 = [0]
    cols2 = [i+3 for i in range(args.days-1)]
    cols3 = cols1 + cols2

    # windows下面由xlsx另存为csv文件，其编码方式为gb2312，而不是utf-8
    df = pd.read_csv(args.src_file, nrows=10766, usecols=cols3, encoding='gb2312')

    # 获取文件行数，列数
    rows, cols = df.shape[:2]                                      

     # 只选取混凝土日均温度这一行，每隔5行取1行
    rows_target = [i for i in range(rows) if i % 5 == 1]          
    df2 = df.iloc[rows_target, :]   
    rows, cols = df2.shape[:2]                                      
    print("Bin numbers: {} \nDays: {} ".format(rows, cols))                       
    # print(df2)

    # 接下来就处理args.df了
    args.df = df2
    return df2


def interpolation(args):
    """
    由于pandas的插值功能，只支持在列方向做插值;
    而混凝土温度数据应该在行方向（日期方向)做插值才是合理的;
    因而先将原来的dataframe数据转置，插值后，再转置回来
    """
    df = args.df
    rows, cols = df.shape

    # 转置后的dataframe
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)



    return



def main(args):
    # 先从原始文件中，读取所需要的行列数据
    process(args)                

    # 每个仓的时间序列长度不一样，需要做插值处理成一样长
    interpolation(args)    

    return




if __name__ == "__main__":

    # 缺失数据的插值方法，其中akima得到的曲线平滑
    # 缺失值主要集中在后段，默认使用nearest最近邻插值
    interp_methods = ["linear", "quadratic", "akima", "nearest"]

    # 参数解析部分，已设置默认值
    parser = argparse.ArgumentParser(description="Process original data.")
    parser.add_argument("--src_file", type=str, default="./data/table_v1.csv", help="source file to be processed")
    parser.add_argument("--dst_file", type=str, default="./data/table_v2.csv", help="destination file")
    parser.add_argument("--days", type=int, default=300, help="读取收仓日期之后的天数")
    parser.add_argument("--interp_method", type=str, default="nearest", choices=interp_methods, help="缺失数据的插值方法，需要使用scipy")
    args = parser.parse_args()
    
    main(args)