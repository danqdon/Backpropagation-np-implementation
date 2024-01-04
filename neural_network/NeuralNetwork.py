import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input):
        result = input
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            activation = X
            activations = [X]  # Lista de todas las activaciones
            for layer in self.layers:
                activation = layer.forward(activation)
                activations.append(activation)

            # Backward pass
            loss_derivative = self.loss_derivative(activations[-1], y)
            for i in reversed(range(len(self.layers))):
                loss_derivative = self.layers[i].backward(activations[i], loss_derivative)

            # Registro del progreso
            if epoch % 10 == 0:
                predicted = self.predict(X)
                loss = self.compute_loss(predicted, y)
                print(f'Epoch {epoch}, Loss: {loss}')

    @staticmethod
    def compute_loss(predicted, actual):
        return np.mean((predicted - actual) ** 2)

    @staticmethod
    def loss_derivative(predicted, actual):
        return 2 * (predicted - actual) / actual.size
