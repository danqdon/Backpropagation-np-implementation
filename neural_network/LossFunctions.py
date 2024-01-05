import numpy as np
from .Activation import Sigmoid
from .Activation import Softmax

class MSE():
    def compute_loss(self,predicted, actual):
        return np.mean((predicted - actual) ** 2)

    def loss_derivative(self,predicted, actual):
        return 2 * (predicted - actual) / actual.size


class CrossEntropy():

    def __init__(self,activation):
        self._layer = activation

    def compute_loss(self,predicted, actual):
        if isinstance(self._layer,Sigmoid):
            predicted = np.clip(predicted, 1e-15, 1 - 1e-15)  # Para estabilidad numérica
            return -np.mean(actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted))
        else: #Si no, asumimos que es una softmax
            predicted = np.clip(predicted, 1e-15, 1 - 1e-15)  # Para estabilidad numérica
            return -np.mean(np.sum(actual * np.log(predicted), axis=1))

    def loss_derivative(self,predicted, actual):
        return predicted - actual
