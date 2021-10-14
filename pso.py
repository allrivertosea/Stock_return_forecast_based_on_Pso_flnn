import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import copy
import random
import matplotlib.pyplot as plt
from particle import Particle

class Population:
    def __init__(self, s, c1, c2, activation = 0):
        self.s = s
        self.c1 = c1
        self.c2 = c2
        # self.c = c1 + c2
        # self.theta = 2 / (self.c - 2 + np.sqrt(np.square(self.c) - 4 * self.c))
        self.activation = activation
        self.Vmax = 1
        self.w = 0.729
        self.w_max = 0.9
        self.w_min = 0.4

    def init_pop(self, d):
        self.population = []
        for i in range(0, self.s):
            c = Particle(d, activation = self.activation)
            self.population.append(c)

    def train(self, X, y, X_valid, y_valid, epochs=1000):
        best_particle = None
        best_fitness = -1
        best_particle_valid = None
        best_fitness_valid = -1

        d = X.shape[1]
        self.init_pop(d)

        for e in range(epochs):
            for p in self.population:
                fitness = p.compute_fitness(X, y)

                if fitness > p.best_fitness:
                    p.best_fitness = fitness
                    p.pbest = copy.deepcopy(p.x)

                if p.best_fitness > best_fitness:
                    best_fitness = fitness
                    best_particle = copy.deepcopy(p)

            valid_fitness = best_particle.compute_fitness(X_valid, y_valid)

            if valid_fitness > best_fitness_valid:
                best_fitness_valid = valid_fitness
                best_particle_valid = copy.deepcopy(best_particle)

            # print("> Epoch {0}: Best fitness {1}   -   Best valid fitness {2}".format(e + 1, best_fitness, best_fitness_valid))

            for p in self.population:
                # w = (1 - e / epochs) * (self.w_max - self.w_min) + self.w_min
                # v_new = self.theta * (p.v + self.c1 * np.random.uniform(low=0,high=1,size=(p.n, 1)) * (p.pbest - p.x) + self.c2 *  np.random.uniform(low=0,high=1,size=(p.n, 1)) * (best_particle.pbest - p.x))
                v_new = self.w * p.v + self.c1 * np.random.uniform(low=0 ,high=1 ,size=(p.n + 1, 1)) * \
                            (p.pbest - p.x) + self.c2 * np.random.uniform(low=0, high=1, size=(p.n + 1, 1)) * (
                                    best_particle.pbest - p.x)

                v_new[v_new > self.Vmax] = self.Vmax
                v_new[v_new < -self.Vmax] = -self.Vmax

                x_new = p.x + v_new

                p.v = copy.deepcopy(v_new)
                p.x = copy.deepcopy(x_new)

        return best_particle_valid