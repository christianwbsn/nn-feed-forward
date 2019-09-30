"""
Module implementing model of neural net
"""

from src.layers.feedforward import FeedForward
from src.layers.activation import Sigmoid
from src.layers.loss import MeanSquaredError
import numpy as np

class Model():
    def __init__(self, nb_nodes, hidden_layer, momentum=0.9, learning_rate=0.01):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocity = 0
        if (len(nb_nodes)-2) != hidden_layer:
            raise ValueError("hidden layer size mismatch with nb_nodes")

        if (len(nb_nodes)-2 > 10 or hidden_layer > 10):
            raise ValueError("maximum hidden layer is 10")
        # Create the fully connected layers
        self.ff = []
        for i in range(len(nb_nodes) - 1):
            self.ff.append(FeedForward(nb_nodes[i], nb_nodes[i + 1]))

        # Instantiate the activation function
        self.sigmoid = Sigmoid()

        # Instantiate the loss / cost function
        self.loss_func = MeanSquaredError()


    def __call__(self, x):
        # A variable to record intermediate output
        self.transitional = []

        # Append the input as one of intermediate output
        self.transitional.append(x)

        # Do forward pass and record each intermediate output in self.transitional
        for layer in self.ff:
            self.transitional.append(layer(self.transitional[-1]))
            self.transitional.append(self.sigmoid(self.transitional[-1]))

        # Return the last output of the last layer
        return self.transitional[-1]

    def back_prop(self, gold):
        for l in range(len(self.ff) - 1, -1, -1):
            # If it's the last layer, compute the gradient of the loss function
            if l == len(self.ff) - 1:
                layer_grad = self.loss_func.calc_grad(gold, self.transitional[2 * (l + 1)])
            # Else, compute the gradient of its successive layer based on its successive activation gradient
            else:
                layer_grad = self.ff[l + 1].calc_grad(activation_grad)

            # Compute the gradient of this layer's activation function, based on its successive layer's gradient
            activation_grad = self.sigmoid.calc_grad(self.transitional[2 * (l + 1) - 1], layer_grad)

            # Based on those gradients, correct the weights
            for o in range(self.ff[l].weight.shape[0]):
                for i in range(self.ff[l].weight.shape[1]):
                    # Add momentum here
                    self.velocity = self.momentum * self.velocity + (1 - self.momentum) * activation_grad[o] * np.mean(self.transitional[2 * (l + 1) - 2], axis = 0)[i]
                    self.ff[l].weight[o, i] -= self.lr * self.velocity
