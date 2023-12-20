import numpy as np
import Utils as u
from Value import Value

class NeuralNetwork:
    def __init__(self, *layer):
        self.layers= list(layer)
        self.train_acc = []
        self.train_loss = []

    def __feed_forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def __update_weights(self,alpha):
        for layer in self.layers:
            layer.update_weights(alpha)

    def train(self,x,Y, alpha=0.01, epoch=10):
        self.train_acc = []
        self.train_loss = []
        for j in range(epoch):
            l = []

            for xi, yi in zip(x, Y):
            # Convert inputs and labels to Value objects
                xi = [Value(xii) for xii in xi]
                yi = [Value(yii) for yii in yi]

            out = self.__feed_forward(xi)
            loss_value = u.cross_entropy_loss(out, yi)
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

    def predict(self, input_data):
        return [self.__feed_forward(x) for x in input_data]


