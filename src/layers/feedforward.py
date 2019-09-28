from numpy.random import rand

class FeedForward():
    def __init__(self, input_dim, output_dim):
        self.weight = rand(output_dim, input_dim)

    def __call__(self, x):
        """
        Apply weigths to input.

        Takes a List of size input_dim
        Returns a List of size output_dim
        """
        # TODO: handle batch
        # TODO: Check shape of x
        return self.weight.dot(x)

    def calc_grad(self, succ_grad): # For computing error term
        """
        Calculate the derivative of each the output with respect to its input (not to its weight) 
        multiplied by all of its successive gradient
        
        Takes a List of size output_dim (the gradient of its successive layer)
        Returns a List of size input_dim (the resulting gradient)
        """
        # TODO: handle batch
        return [sum(succ_grad[j] * self.weight[j, i] for j in range(self.weight.shape[0])) for i in range(self.weight.shape[1])]