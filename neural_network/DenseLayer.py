import numpy as np
from .Layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / input_units)
        self.biases = np.zeros(output_units)
        self._learning_rate = learning_rate

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        # grad_output es el gradiente del costo respecto a la salida de esta capa
        grad_input = np.dot(grad_output, self.weights.T)

        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        self.weights -= self.learning_rate * grad_weights
        self.biases -= self.learning_rate * grad_biases

        return grad_input

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError("El learning rate debe ser mayor que 0")
        self._learning_rate = value
