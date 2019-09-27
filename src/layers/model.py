from feedforward import FeedForward
from activation import Sigmoid
from loss import MeanSquaredError

class Model():
    def __init__(self):
        self.lr = 0.1

        self.ff1 = FeedForward(5, 4)
        self.ff2 = FeedForward(4, 3)
        self.sigmoid = Sigmoid()

        self.loss_func = MeanSquaredError()

    def __call__(self, x):
        self.x = x
        self.y1 = self.ff1(self.x)
        self.y2 = self.sigmoid(self.y1)
        self.y3 = self.ff2(self.y2)
        self.y4 = self.sigmoid(self.y3)

        print(self.loss_func([1, 1, 1], self.y4))

        return self.y4

    def back_prop(self):
        loss_grad = self.loss_func.calc_grad([1, 1, 1], self.y4)
        y4_grad = self.sigmoid.calc_grad(self.y3, loss_grad)

        # correct ff2
        for o in range(self.ff2.weight.shape[0]):
            for i in range(self.ff2.weight.shape[1]):
                self.ff2.weight[o, i] -= self.lr * y4_grad[o] * self.y2[i]

        y3_grad = self.ff2.calc_grad(y4_grad)
        y2_grad = self.sigmoid.calc_grad(self.y2, y3_grad)

        # correct ff1
        for o in range(self.ff1.weight.shape[0]):
            for i in range(self.ff1.weight.shape[1]):
                self.ff1.weight[o, i] -= self.lr * y2_grad[o] * self.x[i]

if __name__ == "__main__":
    model = Model()
    for _ in range(100):
        print(model([1, 1, 1, 1, 1]))
        model.back_prop()
        print()
    