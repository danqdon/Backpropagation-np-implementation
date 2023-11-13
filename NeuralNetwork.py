import numpy as np
import matplotlib.pyplot as plt
from Utils import loss,sigmoid
from Backpropagation import back_prop

class NeuralNetwork:

    def __init__(self,input_size,hidden_size,output_size,seed):
        np.random.seed = seed
        self.seed = seed
        self.w1 = np.random.randn(input_size, hidden_size)
        self.w2 = np.random.randn(hidden_size, output_size)
        self.train_acc = []
        self.train_loss = []

    def train(self, x, Y, alpha=0.01, epoch=10):
        # No me convence mucho esto que he hecho pero es para que persista
        # para luego visualizarlo.
        self.train_acc = []
        self.train_loss = []
        for j in range(epoch):
            l = []
            for i in range(len(x)):
                out = self.__feed_forward(x[i],self.w1, self.w2)
                l.append((loss(out, Y[i])))
                self.w1, self.w2 = back_prop(x[i], Y[i], self.w1, self.w2, alpha)
            print("epochs:", j + 1, "======== acc:", (1 - (sum(l) / len(x))) * 100)
            self.train_acc.append((1 - (sum(l) / len(x))) * 100)
            self.train_loss.append(sum(l) / len(x))


    def predict(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        Out = self.__feed_forward(x, self.w1, self.w2)
        maxm = 0
        k = 0
        for i in range(len(Out[0])):
            if (maxm < Out[0][i]):
                maxm = Out[0][i]
                k = i
        if (k == 0):
            print("Image is of letter D.")
        elif (k == 1):
            print("Image is of letter J.")
        else:
            print("Image is of letter C.")
        plt.imshow(x.reshape(5, 6))
        plt.show()

    def __feed_forward(x, w1, w2):
        # hidden
        z1 = x.dot(w1)  # input from layer 1
        a1 = sigmoid(z1)  # out put of layer 2
        # Output layer
        z2 = a1.dot(w2)  # input of out layer
        a2 = sigmoid(z2)  # output of out layer
        return (a2)
