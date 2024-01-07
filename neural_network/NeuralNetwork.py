from .LossFunctions import MSE, CrossEntropy
from neural_network import DenseLayer
from .Activation import Sigmoid, Softmax

class NeuralNetwork:
    def __init__(self, *layers):
        self.layers = list(layers)
        self.last_activation_layer = None
        if layers and isinstance(layers[-1], (Sigmoid, Softmax)):
            self.last_activation_layer = layers[-1]

    def predict(self, input):
        result = input
        for layer in self.layers:
            result = layer.forward(result)
        return result

    def train(self, X, y, epochs, learning_rate, loss='mse'):
        self.__set_learning_rate(learning_rate)
        loss_history = []  # Historial de pérdida
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

            # Registrar el progreso
            predicted = self.predict(X)
            loss_value = self.compute_loss(predicted, y, loss)
            loss_history.append(loss_value)  # Guardar la pérdida para cada época

        return loss_history  # Devolver el historial de pérdida

    def compute_loss(self, predicted, actual, loss):
        if loss == 'mse':
            return MSE().compute_loss(predicted, actual)
        else:
            return CrossEntropy(self.last_activation_layer).compute_loss(predicted, actual)

    def loss_derivative(self, predicted, actual, loss):
        if loss == 'mse':
            return MSE().loss_derivative(predicted, actual)
        else:
            return CrossEntropy(self.last_activation_layer).loss_derivative(predicted, actual)

    def __set_learning_rate(self, new_lr):
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                layer.learning_rate = new_lr
