# nn-feed-forward
IF4074 - Advanced Machine Learning

## Mathematical Formulation
We want to correct the weight of our model to produce the smallest error possible. To do that, we must use calculus: find the derivative of our loss function with respect to the specific weight that we want to correct, and then substract that weight proportional to the derivative. 

The derivative must be calculated with a chain rule. In the code, `Layer.calc_grad()` is used to calculate a single term in the chain rule. 
