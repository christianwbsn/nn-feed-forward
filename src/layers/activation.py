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

    def __call__(self, X):
        """
        Calculate the elementwise sigmoid of a List

        Takes a `List` of any size
        Returns a `List` of the same size
        """
        X = np.array(X)
        return 1 / (1 + np.exp(-X))

    def calc_grad(self, X, succ_grad):
        """
        Calculate the derivative of sigmoid with respect to each element in x 
        multiplied by all of its successive gradient

        Takes:
            x, a `List` of any size. The input
            succ_grad, a `List` with the same size as `x`. The gradient of its successive layer
        Returns a List with the same size as x. The resulting gradient
        """
        
        X = np.array(X)
        if X.shape[1] != len(succ_grad):
            raise ValueError("Grad size mismatch")

        cur_grad = np.mean(self(X) * (1 - self(X)), axis = 0)
        return cur_grad * succ_grad