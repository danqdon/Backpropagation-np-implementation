import numpy as np
from LossFunctions import MSE, CrossEntropy
from neural_network import DenseLayer


class NeuralNetwork:
    def __init__(self, *layers):
        self.layers = list(layers)

    def predict(self, input):
        result = input
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def train(self, X, y, epochs, learning_rate,loss='mse'):
        self.__set_learning_rate(learning_rate)
        for epoch in range(epochs):
            # Forward pass
            activation = X
            activations = [X]  # Lista de todas las activaciones
            for layer in self.layers:
                activation = layer.forward(activation)
                activations.append(activation)

            # Backward pass
            loss_derivative = self.loss_derivative(activations[-1], y, loss)
            for i in reversed(range(len(self.layers))):
                loss_derivative = self.layers[i].backward(activations[i], loss_derivative)

            # Registro del progreso
            if epoch % 10 == 0:
                predicted = self.predict(X)
                loss = self.compute_loss(predicted, y,loss,self.layers[-1])
                print(f'Epoch {epoch}, Loss: {loss}')

    @staticmethod
    def compute_loss(predicted, actual,loss,last_layer):
        if loss == 'mse':
            return MSE().compute_loss(predicted,actual)
        else:
            return CrossEntropy(last_layer).compute_loss(predicted,actual)

    @staticmethod
    def loss_derivative(predicted, actual,loss):
        if loss == 'mse':
            return MSE().loss_derivative(predicted, actual)
        else:
            return CrossEntropy().loss_derivative(predicted, actual)

    def __set_learning_rate(self,new_lr):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.learning_rate = new_lr
