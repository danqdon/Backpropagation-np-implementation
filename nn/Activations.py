from nn.Layer import Layer
import numpy as np
from Value import Value
class Sigmoid(Layer):
    def forward(self, x):
        self.input = x
        return [xi.sigmoid() for xi in x]

    def backprop(self, output_grad):
        input_grad = [0 for _ in self.input]
        for i, grad in enumerate(output_grad):
            self.input[i].grad = grad
            self.input[i].backward()
            input_grad[i] = self.input[i].grad
        return input_grad

    def update_weights(self, alpha):
        pass  # Sigmoid has no weights


class ReLU(Layer):
    def __init__(self):
        self.input = None  # Store the input to the layer

    def forward(self, x):
        self.input = x  # Store the input
        return [xi.relu() for xi in x]

    def backprop(self, output_grad):
        input_grad = [Value(0) for _ in self.input]
        for i, grad in enumerate(output_grad):
            # ReLU derivative: 0 for negative input, 1 for positive input
            self.input[i].grad = grad * (1 if self.input[i].value > 0 else 0)
            self.input[i].backward()
            input_grad[i] = self.input[i].grad
        return input_grad

    def update_weights(self, alpha):
        # ReLU has no weights to update
        pass

