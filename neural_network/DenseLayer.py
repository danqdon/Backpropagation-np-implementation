import numpy as np
from .Layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_units, output_units, optimizer='adam', learning_rate=0.1):
        self.weights = np.random.randn(input_units, output_units) * np.sqrt(2. / input_units)
        self.biases = np.zeros(output_units)
        self.optimizer = optimizer
        self._learning_rate = learning_rate

        # Atributos para RMSProp
        self.rho = 0.9
        self.epsilon = 1e-8
        self.s_grad = np.zeros_like(self.weights)

        # Atributos para SGD con momentum
        self.momentum = 0.9
        self.velocity = np.zeros_like(self.weights)

        #Atributos para ADAM
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m = np.zeros_like(self.weights)  # Primer momento
        self.v = np.zeros_like(self.weights)  # Segundo momento
        self.adam_iteration = 1  # Para corregir los primeros momentos

    def forward(self, input):
        return np.dot(input, self.weights) + self.biases

    def backward(self, input, grad_output):
        # Calcular gradientes
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0) * input.shape[0]

        # Aplicar el optimizador seleccionado
        if self.optimizer == 'adam':
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad_weights
            self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad_weights)

            m_hat = self.m / (1 - self.beta1 ** self.adam_iteration)
            v_hat = self.v / (1 - self.beta2 ** self.adam_iteration)

            self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.biases -= self.learning_rate * grad_biases
            self.adam_iteration += 1

        # Aplicar el optimizador seleccionado
        if self.optimizer == 'sgd_momentum':
            self.velocity = self.momentum * self.velocity - self.learning_rate * grad_weights
            self.weights += self.velocity
            self.biases -= self.learning_rate * grad_biases
        elif self.optimizer == 'rmsprop':
            self.s_grad = self.rho * self.s_grad + (1 - self.rho) * np.square(grad_weights)
            self.weights -= self.learning_rate * grad_weights / (np.sqrt(self.s_grad) + self.epsilon)
            self.biases -= self.learning_rate * grad_biases
        else:
            self.weights -= self.learning_rate * grad_weights
            self.biases -= self.learning_rate * grad_biases

        return grad_input

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        if value <= 0:
            raise ValueError("El learning rate debe ser mayor que 0")
        self._learning_rate = value
