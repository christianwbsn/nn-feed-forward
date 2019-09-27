"""
Module implementing various activation function in neural network
"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(x, axis=None):
    return np.exp(x) / sum(np.exp(x))

class Sigmoid():
    def __init__(self):
        pass

    def __elementwise_sigmoid__(self, x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x):
        return [self.__elementwise_sigmoid__(x_i) for x_i in x]

    def calc_grad(self, x, succ_grad):
        """
        with respect to each element in x
        """
        return [succ_grad[i] * self.__elementwise_sigmoid__(x[i]) * (1 - self.__elementwise_sigmoid__(x[i])) for i in range(len(x))]