import numpy as np
import dill
from layer import Layer
from feedforward import FeedForward
from activation import Sigmoid
from loss import MeanSquaredError


def mini_batch(inputs, targets, batch_size, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        end_idx = min(start_idx + batch_size, inputs.shape[0])
        if shuffle:
            excerpt = indices[start_idx : end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt], targets[excerpt]

class Model():
    def __init__(self, num_nodes, cost_function):
        self.num_layers = len(num_nodes)
        self.num_nodes = num_nodes
        self.layers = []
        self.cost_function = cost_function

        for i in range(self.num_layers):
            if i != self.num_layers-1:
                layer_i = Layer(num_nodes[i], num_nodes[i+1])
            else:
                layer_i = Layer(num_nodes[i], 0)
            self.layers.append(layer_i)

    def fit(self, batch_size, inputs, labels, num_epochs, learning_rate):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        for j in range(num_epochs):
            for batch in mini_batch(inputs, labels, batch_size, shuffle=True):
                self.error = 0
                self.forward_pass(batch[0])
                self.calculate_error(batch[1])
                self.back_prop(batch[1])
            self.error /= batch_size
            print("EPOCH: ", j+1, "/", num_epochs, " - Error: ", self.error)
        dill.dump_session("model.pkl")

    def forward_pass(self, inputs):
        self.layers[0].activations = inputs
        for i in range(self.num_layers-1):
            temp = np.matmul(self.layers[i].activations, self.layers[i].weights_for_layer)
            self.layers[i+1].activations = self.sigmoid(temp)

    def sigmoid(self, layer):
        return np.divide(1, np.add(1, np.exp(np.negative(layer))))

    def calculate_error(self, labels):
        if self.cost_function == "mean_squared":
            self.error += np.mean(np.divide(np.square(np.subtract(labels, self.layers[self.num_layers-1].activations)), 2))
        elif self.cost_function == "cross_entropy":
            self.error += np.negative(np.sum(np.multiply(labels, np.log(self.layers[self.num_layers-1].activations))))

    def back_prop(self, labels):
        targets = labels
        i = self.num_layers-1
        y = self.layers[i].activations
        deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, targets-y)))
        new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
        for i in range(i-1, 0, -1):
            y = self.layers[i].activations
            deltaw = np.matmul(np.asarray(self.layers[i-1].activations).T, np.multiply(y, np.multiply(1-y, np.sum(np.multiply(new_weights, self.layers[i].weights_for_layer),axis=1).T)))
            self.layers[i].weights_for_layer = new_weights
            new_weights = self.layers[i-1].weights_for_layer - self.learning_rate * deltaw
        self.layers[0].weights_for_layer = new_weights

    def predict(self, input):
        dill.load_session("model.pkl")
        self.batch_size = len(input)
        self.forward_pass(input)
        a = self.layers[self.num_layers-1].activations
        a[np.where(a==np.max(a))] = 1
        a[np.where(a!=np.max(a))] = 0
        return a