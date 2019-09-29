from feedforward import FeedForward
from activation import Sigmoid
from loss import MeanSquaredError

class Model():
    def __init__(self, nb_nodes):
        self.lr = 0.005

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
                    self.ff[l].weight[o, i] -= self.lr * activation_grad[o] * self.transitional[2 * (l + 1) - 2][i]

if __name__ == "__main__":
    model = Model([6, 6, 2])
    for _ in range(1000):
        print(model([1, 1, 1, 1, 1, 1]))
        model.back_prop([0.5, 0.5])
    