import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
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

    def train(self, X, y, epochs, learning_rate, loss='mse',patience=None,valid_pc=0.2,min_delta=0.001):
        self.__set_learning_rate(learning_rate)
        early_stopping = False
        loss_history_train = []
        r2_history_train = []

        if patience is not None:
            X, X_valid, y, y_valid = train_test_split(X, y, test_size=valid_pc, random_state=42)
            early_stopping = True
            r2_history_valid = []
            loss_history_valid = []
            best_mse = np.inf
            epochs_no_improve = 0

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
            predicted_train = self.predict(X)
            loss_value = self.compute_loss(predicted_train, y, loss)
            loss_history_train.append(loss_value)
            r2_train = r2_score(y.flatten(), predicted_train.flatten())
            r2_history_train.append(r2_train)
            print(f'Epoch {epoch}, Loss Train(MSE): {loss_value}, R2 Train: {r2_train}')

            if early_stopping:
                predicted_validation = self.predict(X_valid)
                loss_value_valid = self.compute_loss(predicted_validation, y_valid, loss)
                loss_history_valid.append(loss_value_valid)
                r2_valid = r2_score(y_valid.flatten(), predicted_validation.flatten())
                r2_history_valid.append(r2_valid)
                print(f',Loss Valid(MSE): {loss_value_valid}, R2 Valid: {r2_valid}')

                best_mse, epochs_no_improve = self.__early_stopping(loss_value_valid, best_mse, epochs_no_improve,min_delta)

                if epochs_no_improve == patience:
                    print(f'Early stopping: MSE no mejora desde la época {epoch - patience}')
                    break

        metrics = {
            'loss_history_train': loss_history_train,
            'r2_history_train': r2_history_train,
        }

        if early_stopping:
            metrics.update({
                'loss_history_valid': loss_history_valid,
                'r2_history_valid': r2_history_valid
            })

        return metrics

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

    def __early_stopping(self, loss_value_valid, best_mse, epochs_no_improve, min_delta):
        if best_mse - loss_value_valid > min_delta:
            best_mse = loss_value_valid
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        return best_mse, epochs_no_improve



