#!/usr/bin/python3 
# coding: utf-8


def delete_interval_value(src_file, dst_file, skip_rows, skip_columns, min_value, max_value):
    """
    删除[min_value, max_value]之间的值
    src_file: 原始输入文件，要求是一个.csv文件
    dst_file: 输出文件
    skip_rows: 跳过的行数，即不处理前几行
    skip_columns: 跳过的列数，即不处理前几列
    min_value: 删除的最小区间值
    max_value: 删除的最大区间值
    """
    with open(src_file, 'r', encoding='gbk') as fr, open(dst_file, 'w', encoding='gbk') as fw:
        lines = fr.readlines()
        res = []
        for i,line in enumerate(lines):
            if i < skip_rows:
                res.append(line)
            else:
                parts = line.strip().split(",")
                for j in range(len(parts)):
                    if j >= skip_columns and parts[j] != "":
                        value = float(parts[j])
                        if value >= min_value and value <= max_value:
                            parts[j] = ""

                res.append(",".join(k for k in parts) + "\n")

        for line in res:
            fw.writelines(line)
    return



if __name__ == "__main__":

    src_file = "./data/仓内温差数据处理.csv"
    dst_file = "./data/仓内温差数据处理-删除区间后.csv"
    skip_rows = 1
    skip_columns = 1
    min_value = 1.0
    max_value = 2.0
    delete_interval_value(src_file, dst_file, skip_rows, skip_columns, min_value, max_value)
