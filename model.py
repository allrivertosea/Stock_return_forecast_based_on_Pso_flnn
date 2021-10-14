import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from pso import Population
from expand_data import ExpandData

class Model:
    def __init__(self, data_original, train_idx, test_idx, expand_func=0, pop_size=200, c1=2, c2=2,
                 activation=2, data_filename = "pre_return"):
        self.data_original = data_original
        self.train_idx = train_idx
        self.test_idx = test_idx
        self.data = data_original[:train_idx + test_idx + 1, :]
        self.scaler = MinMaxScaler()
        self.expand_func = expand_func
        self.pop_size = pop_size
        self.c1 = c1
        self.c2 = c2
        self.activation = activation
        self.pathsave = r"data\预测模型下载数据\\"
        self.textfilename = "test_pre_return"
        self.filename = "{0}-PSO-FLNN-expand_func_{1}-pop_size_{2}-c1_{3}-c2_{4}-activation_{5}".format(data_filename,  expand_func, pop_size, c1, c2, activation)

    def draw_predict(self):
        plt.figure(2)
        plt.plot(self.real_inverse[:, 0], color='#009FFD', linewidth=2.5)  # 实际周收益
        plt.plot(self.pred_inverse[:, 0], color='#FFA400', linewidth=2.5)  # 预测周收益
        plt.ylabel('周收益率')  # 我的则是周收益
        plt.xlabel('周数')  # 我的是周时间序列
        plt.legend(['Actual', 'Prediction'], loc='upper right')
        # plt.savefig(self.pathsave + self.filenamesave + ".png")
        plt.show()
        plt.close()

    def save_file_csv(self):
        t1 = np.concatenate( (self.pred_inverse, self.real_inverse), axis = 1)
        np.savetxt(self.pathsave + self.filename + ".csv", t1, delimiter=",")

    def write_to_result_file(self):
        with open(self.pathsave + self.textfilename + '.txt', 'a') as file:
            file.write("{0}  -  {1}  -  {2}\n".format(self.filename, self.mae, self.rmse))

    def preprocessing_data(self):

        data, train_idx, test_idx, expand_func = self.data, self.train_idx, self.test_idx, self.expand_func
        data_scale = self.scaler.fit_transform(data)  # 各个样本的特征数据归一化,但是真实收益值也要归一化吗
        data_transform = data_scale[:train_idx + test_idx, :]  # 需要进行拓展的样本，训练集和测试集

        data_x_not_expanded = data_transform[:, :-1]  ##样本的特征值矩阵,归一化后的
        data_y = data_transform[:, [-1]]  # 样本的预测值矩阵，归一化后的

        expand_data_obj = ExpandData(data, train_idx, test_idx, expand_func=expand_func)
        data_expanded_temp = expand_data_obj.process_data()

        data_expanded = []
        #需要改
        for i in range(192):
            temp = []
            for j in range(4):
                for k in range(5):
                    temp.append(data_expanded_temp[k][i][j])
            data_expanded.append(temp)
        data_X = np.concatenate((data_x_not_expanded, data_expanded), axis=1)  # 原特征值加上拓展的特征值形成新的特征值矩阵
        self.X_train, self.X_test, self.y_train, self.y_test = data_X[:train_idx, :], data_X[train_idx:, :], data_y[
                                                                                                             :train_idx,
                                                                                                             :], data_y[
                                                                                                                 train_idx:,
                                                                                                                 :]
        self.X_train_old, self.X_test_old, self.y_train_old, self.y_test_old = data_x_not_expanded[:train_idx,
                                                                               :], data_x_not_expanded[train_idx:,
                                                                                   :], data_y[
                                                                                       :train_idx,
                                                                                       :], data_y[
                                                                                           train_idx:,
                                                                                           :]

    def train(self, epochs=500):
        self.preprocessing_data()

        p = Population(self.pop_size, self.c1, self.c2, activation=self.activation)

        best = p.train(self.X_train, self.y_train, self.X_test, self.y_test, epochs=epochs)

        pred = best.predict(self.X_test)
        X_pred = np.concatenate((self.X_test_old, pred), axis=1)
        X_y_test = np.concatenate((self.X_test_old, self.y_test), axis=1)
        # print(self.y_test)
        temp_pred_inverse = self.scaler.inverse_transform(X_pred)
        temp_real_inverse = self.scaler.inverse_transform(X_y_test)
        self.pred_inverse = []  # 得到预测值的逆转换值
        self.real_inverse = []  # 得到真实值的逆转换值
        for i in temp_pred_inverse:
            self.pred_inverse.append(i[3])
        for i in temp_real_inverse:
            self.real_inverse.append(i[3])
        print(self.real_inverse)
        print(self.pred_inverse)
        print(mean_squared_error(self.real_inverse, self.pred_inverse))  # 得到最后的误差
        return self.pred_inverse
##################
        # self.pred_inverse = self.scaler.inverse_transform(pred)
        # self.real_inverse = self.scaler.inverse_transform(self.y_test)
        #
        # self.mae = mean_absolute_error(self.real_inverse[:, 0], self.pred_inverse[:, 0])
        # self.rmse = np.sqrt(mean_squared_error(self.real_inverse[:, 0], self.pred_inverse[:, 0]))
        #
        # print(self.mae)
        #
        # self.draw_predict()
        #
        # self.write_to_result_file()
        #
        # self.save_file_csv()