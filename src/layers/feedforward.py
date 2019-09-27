from numpy.random import rand

class FeedForward():
    def __init__(self, input_dim, output_dim):
        self.weight = rand(output_dim, input_dim)
        # print(self.weight.shape)
        self.previous_input = None # For computing final gradient
        # TODO: Initialize

    def __call__(self, x):
        # TODO: handle batch
        # TODO: Check shape of x
        return self.weight.dot(x)

    def calc_grad(self, succ_grad): # For computing error term
        """
        With respect to its the input, not to its weight
        """
        return [sum(succ_grad[j] * self.weight[j, i] for j in range(self.weight.shape[0])) for i in range(self.weight.shape[1])]