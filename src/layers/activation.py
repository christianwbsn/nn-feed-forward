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
        """
        Calculate the elementwise sigmoid of a List

        Takes a `List` of any size
        Returns a `List` of the same size
        """
        # TODO: handle batch
        return [self.__elementwise_sigmoid__(x_i) for x_i in x]

    def calc_grad(self, x, succ_grad):
        """
        Calculate the derivative of sigmoid with respect to each element in x 
        multiplied by all of its successive gradient

        Takes:
            x, a `List` of any size. The input
            succ_grad, a `List` with the same size as `x`. The gradient of its successive layer
        Returns a List with the same size as x. The resulting gradient
        """
        # TODO: handle batch
        return [succ_grad[i] * self.__elementwise_sigmoid__(x[i]) * (1 - self.__elementwise_sigmoid__(x[i])) for i in range(len(x))]