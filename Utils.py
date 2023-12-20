import numpy as np
from Value import Value

def sigmoid_prime(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)

def mse_loss(predicted, target):
    # Ensure that the sum starts with a Value object representing 0
    return sum([(p - t) ** 2 for p, t in zip(predicted, target)], Value(0)) / Value(len(predicted))


