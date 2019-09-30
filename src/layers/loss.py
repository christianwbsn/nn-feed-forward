"""
Module implementing loss function
"""
from numpy import array, mean

class MeanSquaredError():
    def __init__(self):
        pass

    def __call__(self, gold, pred):
        """
        Calculate the mean squared error of `pred` against `gold`

        Takes:
            gold, a `List` of any size. The gold standard, the true y
            pred, a `List` with the same size as gold. The predicted y
        Returns a `Float` that indicates the MSE
        """

        # Check dimension equality
        gold = array(gold)
        pred = array(pred)
        if gold.shape != pred.shape:
            raise ValueError("gold and pred is not the same size")

        return mean((gold - pred)**2)

    def calc_grad(self, gold, pred):
        """
        Calculate the derivative of MSE with respect to each `pred`

        Takes:
            gold, a `List` of any size. The gold standard, the true y
            pred, a `List` with the same size as gold. The predicted y
        Return a `List` with the same size as `gold` and `pred` (the gradient)
        """

        gold = array(gold)
        pred = array(pred)
        if gold.shape != pred.shape:
            raise ValueError("gold and pred is not the same size")

        return mean(-2 * (gold - pred), axis = 0)