import numpy as np

class Layer:
    def __init__(self, num_nodes_in_layer, num_nodes_in_next_layer):
        self.num_nodes_in_layer = num_nodes_in_layer
        self.activations = np.zeros([num_nodes_in_layer,1])
        if num_nodes_in_next_layer != 0:
            self.weights_for_layer = np.random.normal(0, 0.001, size=(num_nodes_in_layer, num_nodes_in_next_layer))
        else:
            self.weights_for_layer = None