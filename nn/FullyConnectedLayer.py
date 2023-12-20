from nn.Layer import Layer
from Value import Value
import numpy as np

class FullyConnected(Layer):
    def __init__(self,input_size, output_size):
        self.w = np.array([[Value(w) for w in np.random.randn(input_size)] for _ in range(output_size)], dtype=object)
    def forward(self, x):
        z = [sum([wi * xi for wi, xi in zip(wrow, x)], Value(0)) for wrow in self.w]
        a = [i.sigmoid() for i in z]

        return a

    def backprop(self, output_grad):
        input_grad = np.zeros_like(self.input)
        for i, grad in enumerate(output_grad):
            self.input[i].grad = grad
            self.input[i].backward()  # Propagate gradients backwards
            input_grad[i] = self.input[i].grad
        return input_grad

    def update_weights(self,alpha):
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                self.w[i][j].value -= alpha * self.w[i][j].grad
                self.w[i][j].grad = 0