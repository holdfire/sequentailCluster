#!/usr/bin/python3 
# coding: utf-8
import numpy as np


def stack_table(src_file, dst_file, skip_rows, skip_columns):
    """
    删除[min_value, max_value]之间的值
    src_file: 原始输入文件，要求是一个.csv文件
    dst_file: 输出文件
    skip_rows: 跳过的行数，即不处理前几行
    skip_columns: 跳过的列数，即不处理前几列
    """
    with open(src_file, 'r', encoding='gbk') as fr, open(dst_file, 'w', encoding='gbk') as fw:
        lines = fr.readlines()
        res = []
        for i,line in enumerate(lines):
            if i < skip_rows:
                continue
            else:
                res.append(list(x for x in line.strip().split(","))[skip_columns:])

        res = np.array(res)
        print(res.shape)

        for i in range(res.shape[1]):
            for j in range(res.shape[0]):
                fw.writelines(res[j][i] + "\n")

    return



if __name__ == "__main__":

    src_file = "./data/仓内温差数据处理.csv"
    dst_file = "./data/仓内温差数据处理-堆积数值后.csv"
    skip_rows = 1
    skip_columns = 1
    stack_table(src_file, dst_file, skip_rows, skip_columns)
