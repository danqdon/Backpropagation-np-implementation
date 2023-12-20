import numpy as np
from Value import Value
import Utils as u

class NeuralNetwork:
    def __init__(self, *layers):
        self.layers = list(layers)
        self.train_acc = []
        self.train_loss = []

    def __feed_forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def __backpropagation(self, loss_grad):
        # Start from the last layer and move backwards
        for layer in reversed(self.layers):
            loss_grad = layer.backprop(loss_grad)

    def __update_weights(self, alpha):
        for layer in self.layers:
            layer.update_weights(alpha)

    def train(self, x, Y, alpha=0.01, epoch=10):
        self.train_acc = []
        self.train_loss = []
        for j in range(epoch):
            epoch_loss = 0

            for xi, yi in zip(x, Y):
                # Convert inputs and labels to Value objects
                xi = [Value(xii) for xii in xi]
                yi = [Value(yii) for yii in yi]

                out = self.__feed_forward(xi)
                loss_value = u.cross_entropy_loss(out, yi)
                epoch_loss += loss_value.value

                # Perform backpropagation
                self.__backpropagation([loss_value])

                # Update weights
                self.__update_weights(alpha)

            avg_epoch_loss = epoch_loss / len(x)
            epoch_acc = (1 - avg_epoch_loss) * 100  # Simplified accuracy calculation
            print(f"Epoch: {j + 1}, Accuracy: {epoch_acc:.2f}%")
            self.train_acc.append(epoch_acc)
            self.train_loss.append(avg_epoch_loss)

    def predict(self, input_data):
        predictions = []
        for x in input_data:
            x = [Value(xi) for xi in x]  # Convert inputs to Value objects
            output = self.__feed_forward(x)
            predictions.append(output[-1].value)  # Assuming the output is the last element
        return predictions
