import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, n, activation = 0):
        self.n = n
        self.x = np.random.uniform(low = -1, high = 1, size=(n + 1, 1))
        ## 粒子位置，由权重向量变量组成，变量范围，-1,1，神经网络一般是这样，n+1表示有偏置
        self.v = np.zeros((n + 1, 1))
        self.pbest = np.copy(self.x)
        self.best_fitness = -1
        self.activation = activation

    def activation_output(self, z):
        if self.activation == 0:
            a = z
        elif self.activation == 1:
            a = self.tanh(z)
        elif self.activation == 2:
            a = self.relu(z)
        elif self.activation == 3:
            a = self.elu(z)
        return a

    def compute_fitness(self, X, y):
        x, b = self.x[:-1, :], self.x[[-1], :]
        z = np.dot(X, x) + b

        a = self.activation_output(z)

        mae = mean_absolute_error(y, a)
        self.fitness = 1.0 / mae
        return self.fitness

    def tanh(self, x):
        exp_plus = np.exp(x)
        exp_minus = np.exp(-x)

        return (exp_plus - exp_minus) / (exp_plus + exp_minus)

    def relu(self, x):
        return np.where(x < 0, 0, x)

    def elu(self, x, alpha = 0.5):
        return np.where(x < 0, alpha * (np.exp(x) - 1), x)

    def predict(self, X):
        x, b = self.x[:-1, :], self.x[[-1], :]
        z = np.dot(X, x) + b
        a = self.activation_output(z)
        return a
