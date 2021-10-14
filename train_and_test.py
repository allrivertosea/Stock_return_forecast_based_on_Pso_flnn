import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
import copy
import random
import matplotlib.pyplot as plt
import os
from model import Model

file_dir = r'F:\mygithub\Pso_flnn_stock_return_forecast\data\100只股票三维特征数据集合'
def file_name(dir):
    for root, dirs, files in os.walk(dir):
        file_list_temp = files  # 当前路径下所有非目录子文件
    return file_list_temp
file_list = file_name(file_dir)
print(file_list)
security_code = []
for i in file_list:
    i = i[0:9]
    security_code.append(i)#这时140股票的代码集合
print(security_code)
#现仅用600345.SH进行测试，全测试可令stock = security_code
stock = ['600345.SH.csv']

for i in range(len(stock)):
    csv_file = r"F:\mygithub\Pso_flnn_stock_return_forecast\data\100只股票三维特征数据集合\%s"%(stock[i])
    csv_data = pd.read_csv(csv_file)
    print(stock[i])
    dt = csv_data.drop(['Unnamed: 0','周数'],axis=1)
    train_idx = 140
    test_idx = 52
    model = Model(dt.values, train_idx, test_idx)  # .values把dataframe转化为矩阵
    pre_value = pd.DataFrame(model.train())
    pre_value.to_csv(r'F:\mygithub\Pso_flnn_stock_return_forecast\data\100只股票预测收益集合\%s.csv'%(file_list[i][0:9]))
