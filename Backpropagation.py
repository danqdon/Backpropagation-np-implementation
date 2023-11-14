from Utils import sigmoid, sigmoid_prime, gradient_descent
import numpy as np
def back_prop(x, y, w1, w2, alpha):
    # Forward pass
    input_to_hidden = x.dot(w1)
    hidden_activation = sigmoid(input_to_hidden)
    hidden_to_output = hidden_activation.dot(w2)
    output_activation = sigmoid(hidden_to_output)

    # Compute the error at the output
    derivative_output_error = output_activation - y

    # Compute the derivative of the sigmoid function at the output layer
    derivative_output_activation = sigmoid_prime(hidden_to_output)

    # Compute the gradient for the output layer weights
    gradient_w2_weights = hidden_activation.T.dot(derivative_output_activation * derivative_output_error)

    # Compute the error at the hidden layer by backpropagating the output error through the weights
    hidden_error = (derivative_output_error * derivative_output_activation).dot(w2.T)

    # Compute the derivative of the sigmoid function at the hidden layer
    derivative_hidden_activation = sigmoid_prime(input_to_hidden)

    # Compute the gradient for the hidden layer weights
    gradient_w1_weights = x.T.dot(derivative_hidden_activation * hidden_error)

    # Update weights with the computed gradients

    w1,w2 = gradient_descent(w1,w2,gradient_w1_weights,gradient_w2_weights,alpha)

    return w1, w2