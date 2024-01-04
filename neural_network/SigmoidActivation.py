import numpy as np
from .Layer import Layer

class SigmoidActivation(Layer):
    def forward(self, input):
        return 1 / (1 + np.exp(-input))

    def backward(self, input, grad_output):
        sigmoid_forward = self.forward(input)
        return grad_output * sigmoid_forward * (1 - sigmoid_forward)
