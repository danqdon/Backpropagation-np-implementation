import numpy as np
import matplotlib.pyplot as plt
from Utils import mse_loss
from Value import Value

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seed):
        np.random.seed(seed)
        self.w1 = np.array([[Value(w) for w in np.random.randn(input_size)] for _ in range(hidden_size)], dtype=object)
        self.w2 = np.array([[Value(w) for w in np.random.randn(hidden_size)] for _ in range(output_size)], dtype=object)

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
        # Assuming x is already a list of Value objects
        z1 = [sum([wi * xi for wi, xi in zip(wrow, x)], Value(0)) for wrow in self.w1]
        a1 = [z.sigmoid() for z in z1]  # Apply sigmoid to each Value object in z1

        # Prepare for the next layer
        z2 = [sum([wi * ai for wi, ai in zip(wrow, a1)], Value(0)) for wrow in self.w2]
        a2 = [z.sigmoid() for z in z2]  # Apply sigmoid to each Value object in z2

        return a2

    def __update_weights(self, alpha):
        # Update weights using gradients calculated by autodiff
        for i in range(len(self.w1)):
            for j in range(len(self.w1[i])):
                self.w1[i][j].value -= alpha * self.w1[i][j].grad
                self.w1[i][j].grad = 0  # Reset gradient after update

        for i in range(len(self.w2)):
            for j in range(len(self.w2[i])):
                self.w2[i][j].value -= alpha * self.w2[i][j].grad
                self.w2[i][j].grad = 0  # Reset gradient after update

    def predict(self, input_data):
        return [self.__feed_forward(x) for x in input_data]
