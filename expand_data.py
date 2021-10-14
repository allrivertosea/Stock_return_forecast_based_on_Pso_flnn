import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt

class ExpandData:
    def __init__(self, data, train_idx, test_idx, expand_func = 0):
        self.data = data
        self.scaler = MinMaxScaler()
        self.train_idx = train_idx
        self.test_idx = test_idx
        # self.sliding = sliding
        self.expand_func = expand_func#expand_func = 0，使用切比雪夫拓展

    def expand_data(self):
        if self.expand_func == 0:
            return self.chebyshev()
        elif self.expand_func == 1:
            return self.powerseries()
        elif self.expand_func == 2:
            return self.laguerre()
        elif self.expand_func == 3:
            return self.legendre()

    def legendre(self):
        data = self.data

        x1 = data
        x2 = 3 / 2 * np.power(data, 2) - 1 / 2
        x3 = 1 / 3 * (5*data*x2 - 2*x1)
        x4 = 1 / 4 * (7*data*x3 - 3*x2)
        x5 = 1 / 5 * (9*data*x4 - 4*x3)
        x6 = 1 / 6 * (11*data*x5 - 5*x4)

        return [x2, x3, x4, x5, x6] 

    def laguerre(self):
        data = self.data

        x1 = -data + 1
        x2 = np.power(data, 2) / 2 - 2 * data + 1
        x3 = 1 / 3 * ((5 - data)*x2 - 2*x1)
        x4 = 1 / 4 * ((7 - data)*x3 - 3*x2)
        x5 = 1 / 5 * ((9 - data)*x4 - 4*x3)
        x6 = 1 / 6 * ((11 - data)*x5 - 5*x4)

        return [x2, x3, x4, x5, x6]
    
    def chebyshev(self):
        data = self.data

        x1 = data
        x2 = 2 * np.power(data, 2) - 1
        x3 = 4 * np.power(data, 3) - 3 * data
        x4 = 8 * np.power(data, 4) - 8 * np.power(data, 2) + 1
        x5 = 2 * data * x4 - x3
        x6 = 2 * data * x5 - x4

        return [x2, x3, x4, x5, x6]
    
    def powerseries(self):
        data = self.data

        x1 = data
        x2 = np.power(data, 2)
        x3 = np.power(data, 3)
        x4 = np.power(data, 4)
        x5 = np.power(data, 5)
        x6 = np.power(data, 6)

        return [x2, x3, x4, x5, x6]

    def scale(self, expanded_data):
        scale_expanded_data = []

        for ed in expanded_data:
            scale_expanded_data.append(self.scaler.fit_transform(ed))

        return scale_expanded_data

    def process_data(self):

        train_idx, test_idx = self.train_idx, self.test_idx
        expanded_data = self.expand_data()  # 对data进行切比雪夫拓展，由1个变为5个,每个样本的特征值都由1个变为5个，目前是4013x5，原来是4103x1
        scale_data = self.scale(expanded_data)
        data_expanded = scale_data
        return data_expanded  # 得到140支股票该周的特征值矩阵（标准化后的了）
