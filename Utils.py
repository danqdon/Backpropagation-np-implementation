import numpy as np

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def loss(out, Y):
    s = (np.square(out - Y))
    s = np.sum(s) / len(Y)  # Make sure it's uppercase 'Y'
    return s