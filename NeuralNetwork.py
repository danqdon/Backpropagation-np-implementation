import numpy as np
import matplotlib.pyplot as plt
from Utils import loss, sigmoid, sigmoid_prime, gradient_descent
from Backpropagation import back_prop

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
            # Iterating over each example
            for xi, yi in zip(x, Y):
                out = self.__feed_forward(xi)
                l.append(loss(out, yi))
                self.__back_prop(xi, yi, alpha)
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
        class_labels = ["letter D", "letter J", "letter C"]
        for i in predicted_class:
            print(f"Image is of {class_labels[i]}.")
        plt.imshow(x.reshape(5, 6))
        plt.show()

    def __feed_forward(self, x):
        # Hidden layer
        z1 = x.dot(self.w1)
        a1 = sigmoid(z1)
        # Output layer
        z2 = a1.dot(self.w2)
        a2 = sigmoid(z2)
        return a2

    def __back_prop(self, x, y, alpha):
        # Forward pass
        input_to_hidden = x.dot(self.w1)
        hidden_activation = sigmoid(input_to_hidden)
        hidden_to_output = hidden_activation.dot(self.w2)
        output_activation = sigmoid(hidden_to_output)

        # Compute the error at the output
        derivative_output_error = output_activation - y

        # Compute the derivative of the sigmoid function at the output layer
        derivative_output_activation = sigmoid_prime(hidden_to_output)

        # Compute the gradient for the output layer weights
        gradient_w2_weights = hidden_activation.T.dot(derivative_output_activation * derivative_output_error)

        # Compute the error at the hidden layer by backpropagating the output error through the weights
        hidden_error = (derivative_output_error * derivative_output_activation).dot(self.w2.T)

        # Compute the derivative of the sigmoid function at the hidden layer
        derivative_hidden_activation = sigmoid_prime(input_to_hidden)

        # Compute the gradient for the hidden layer weights
        gradient_w1_weights = x.T.dot(derivative_hidden_activation * hidden_error)

        # Update weights with the computed gradients

        self.w1, self.w2 = gradient_descent(self.w1, self.w2, gradient_w1_weights, gradient_w2_weights, alpha)

