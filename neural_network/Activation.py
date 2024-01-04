import numpy as np
from .Layer import Layer

class ReLU(Layer):
    def forward(self, input):
        return np.maximum(0, input)

    def backward(self, input, grad_output):
        # ReLU derivative is 0 when input is less than 0, 1 otherwise
        relu_grad = input > 0
        return grad_output * relu_grad

class Sigmoid(Layer):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        sigmoid_forward = self.forward(input)
        return grad_output * sigmoid_forward * (1 - sigmoid_forward)
