from feedforward import FeedForward
from activation import Sigmoid
from loss import MeanSquaredError
import numpy as np
import pandas as pd


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
    def __init__(self, nb_nodes, momentum):
        self.lr = 0.0001
        self.momentum = 0.9
        self.velocity = 0

        # Create the fully connected layers
        self.ff = []
        for i in range(len(nb_nodes) - 1):
            self.ff.append(FeedForward(nb_nodes[i], nb_nodes[i + 1]))

        # Instantiate the activation function
        self.sigmoid = Sigmoid()

        # Instantiate the loss / cost function
        self.loss_func = MeanSquaredError()


    def __call__(self, x):
        # if self.batch_size != np.array(x).shape[0]:
        #     raise ValueError("Batch size mismatch")
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

if __name__ == "__main__":
    num_epoch = 100
    df = pd.read_csv('../../data/processed/dataset.csv')
    df = df.drop(df.columns[[0]], axis=1)
    y_train = df.play[0:11].values
    X_train = df.drop('play', 1)[0:11].values
    X_test = df.drop('play', 1)[11:14].values
    y_test = df.play[11:14].values
    model = Model([6, 6, 1], momentum=0.9)
    iter = 0
    for _ in range(num_epoch):
        for batch in mini_batch(X_train, y_train, 5, shuffle=True):
            iter += 1
            model(batch[0])
            model.back_prop(batch[1].reshape(-1,1))
    print(iter)
    print(model(X_test))
