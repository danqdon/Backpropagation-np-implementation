import numpy as np
import matplotlib.pyplot as plt
from Utils import loss, sigmoid, sigmoid_prime, gradient_descent

class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size, seed):
        np.random.seed(seed)
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.train_acc = []
        self.train_loss = []

    def train(self, x, Y, alpha=0.01, epoch=10):
        self.train_acc = []
        self.train_loss = []
        for j in range(epoch):
            l = []

            for xi, yi in zip(x, Y):
                out = self.__feed_forward(xi)
                l.append(loss(out, yi))
                self.__back_propagation(xi, yi, alpha)

            epoch_loss = sum(l) / len(x)
            epoch_acc = (1 - epoch_loss) * 100
            print(f"Epoch: {j + 1}, Accuracy: {epoch_acc:.2f}%")
            self.train_acc.append(epoch_acc)
            self.train_loss.append(epoch_loss)

    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        Out = self.__feed_forward(x)
        predicted_class = np.argmax(Out, axis=1)
        class_labels = ["letter A", "letter E", "letter I","letter O","letter U"]
        for i in predicted_class:
            print(f"Image is of {class_labels[i]}.")
        plt.imshow(x.reshape(8, 8))
        plt.show()

    def __feed_forward(self, x):

        z1 = x.dot(self.w1)
        a1 = sigmoid(z1)

        z2 = a1.dot(self.w2)
        a2 = sigmoid(z2)

        return a2

    def __back_propagation(self, x, y, alpha):

        input_to_hidden = x.dot(self.w1)
        hidden_activation = sigmoid(input_to_hidden)
        hidden_to_output = hidden_activation.dot(self.w2)
        output_activation = sigmoid(hidden_to_output)

        derivative_output_error = output_activation - y

        derivative_output_activation = sigmoid_prime(hidden_to_output)

        gradient_w2_weights = hidden_activation.T.dot(derivative_output_activation * derivative_output_error)

        hidden_error = (derivative_output_error * derivative_output_activation).dot(self.w2.T)

        derivative_hidden_activation = sigmoid_prime(input_to_hidden)

        gradient_w1_weights = x.T.dot(derivative_hidden_activation * hidden_error)

        self.w1, self.w2 = gradient_descent(self.w1, self.w2, gradient_w1_weights, gradient_w2_weights, alpha)

