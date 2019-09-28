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
        if len(gold) != len(pred):
            raise ValueError("gold and pred is not the same size")

        # TODO: handle batch
        return sum([((gold[i] - pred[i])**2)**0.5 for i in range(len(gold))])/len(gold)

    def calc_grad(self, gold, pred):
        """
        Calculate the derivative of MSE with respect to each `pred`

        Takes:
            gold, a `List` of any size. The gold standard, the true y
            pred, a `List` with the same size as gold. The predicted y
        Return a `List` with the same size as `gold` and `pred` (the gradient)
        """

        # Check dimension equality
        if len(gold) != len(pred):
            raise ValueError("gold and pred is not the same size")
        
        # TODO: handle batch
        return [-(gold[i] - pred[i])/(((gold[i]-pred[i])**2)**0.5) for i in range(len(gold))]