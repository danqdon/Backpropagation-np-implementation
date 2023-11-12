import numpy as np

def f_forward(x, w1, w2):
    # hidden
    z1 = x.dot(w1)  # input from layer 1
    a1 = sigmoid(z1)  # out put of layer 2
    # Output layer
    z2 = a1.dot(w2)  # input of out layer
    a2 = sigmoid(z2)  # output of out layer
    return (a2)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def loss(out, Y):
    s = (np.square(out - Y))
    s = np.sum(s) / len(Y)  # Make sure it's uppercase 'Y'
    return s