import numpy as np
import matplotlib.pyplot as plt
from Utils import loss, sigmoid
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
                self.w1, self.w2 = back_prop(xi, yi, self.w1, self.w2, alpha)
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
