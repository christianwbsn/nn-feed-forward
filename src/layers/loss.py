class MeanSquaredError():
    def __init__(self):
        pass

    def __call__(self, gold, pred):
        # TODO: check dimension equality
        return sum([((gold[i] - pred[i])**2)**0.5 for i in range(len(gold))])/len(gold)

    def calc_grad(self, gold, pred):
        """
        Return a vector of derivation of MSE with respect to each pred
        """
        return [-(gold[i] - pred[i])/(((gold[i]-pred[i])**2)**0.5) for i in range(len(gold))]