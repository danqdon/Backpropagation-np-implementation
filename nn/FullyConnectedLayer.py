import numpy as np
from nn.Layer import Layer
from Value import Value
class FullyConnected(Layer):
    def __init__(self, input_size, output_size):
        self.w = np.array([[Value(np.random.randn()) for _ in range(input_size)] for _ in range(output_size)], dtype=object)
        self.input = None  # Store the input to the layer

    def forward(self, x):
        self.input = x  # Store the input
        z = [sum([wi * xi for wi, xi in zip(wrow, x)], Value(0)) for wrow in self.w]
        return z  # Return raw output, activation is separate

    def backprop(self, output_grad):
        input_grad = [Value(0) for _ in self.input]
        for i, grad in enumerate(output_grad):
            self.input[i].grad = grad
            self.input[i].backward()
            input_grad[i] = self.input[i].grad
        return input_grad

    def update_weights(self, alpha):
        for wrow in self.w:
            for w in wrow:
                w.value -= alpha * w.grad
                w.grad = 0
