import numpy as np
import matplotlib.pyplot as plt
from Utils import mse_loss
from Value import Value

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, seed):
        np.random.seed(seed)

        # Initialize weights for each layer
        self.weights = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            w = np.array([[Value(np.random.randn()) for _ in range(layer_sizes[i + 1])] for _ in range(layer_sizes[i])], dtype=object)
            self.weights.append(w)

        self.train_acc = []
        self.train_loss = []

    def train(self, x, Y, alpha=0.01, epoch=10):
        self.train_acc = []
        self.train_loss = []
        for j in range(epoch):
            l = []

            for xi, yi in zip(x, Y):
                # Convert inputs and labels to Value objects
                xi = [Value(xii) for xii in xi]
                yi = [Value(yii) for yii in yi]

                out = self.__feed_forward(xi)
                loss_value = mse_loss(out, yi)
                l.append(loss_value.value)

                # Perform backpropagation
                for o in out:
                    o.grad = 0  # Reset gradients to zero
                loss_value.grad = 1  # Seed the gradient for backpropagation
                loss_value.backward()  # Backward pass to compute gradients

                # Update weights
                self.__update_weights(alpha)

            epoch_loss = sum(l) / len(x)
            epoch_acc = (1 - epoch_loss) * 100  # This is a simplified accuracy calculation
            print(f"Epoch: {j + 1}, Accuracy: {epoch_acc:.2f}%")
            self.train_acc.append(epoch_acc)
            self.train_loss.append(epoch_loss)

    def __feed_forward(self, x):
        activation = x
        for i, w in enumerate(self.weights):
            z = [sum([wi * ai for wi, ai in zip(wrow, activation)], Value(0)) for wrow in w]
            if i < len(self.weights) - 1:
                # ReLU for hidden layers
                activation = [z_i.relu() for z_i in z]
            else:
                # Sigmoid for output layer
                activation = [z_i.sigmoid() for z_i in z]
        return activation

    def __update_weights(self, alpha):
        # Update weights for each layer
        for w_layer in self.weights:
            for i in range(len(w_layer)):
                for j in range(len(w_layer[i])):
                    w_layer[i][j].value -= alpha * w_layer[i][j].grad
                    w_layer[i][j].grad = 0

    def predict(self, input_data):
        raw_predictions = []
        for x in input_data:
            # Convert input to Value objects and perform feedforward
            x_values = [Value(xi) for xi in x]
            output = self.__feed_forward(x_values)
            raw_predictions.append(output[-1].value)  # Get the raw output value
        return raw_predictions

