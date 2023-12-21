import numpy as np

class Value:
    def __init__(self, value, children=None, op=None):
        self.value = value
        self.grad = 0
        self.children = children or []
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.value + other.value, children=[(1, self), (1, other)], op="+")

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.value - other.value, children=[(1, self), (-1, other)], op="-")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.value * other.value, children=[(other.value, self), (self.value, other)], op="*")

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Power must be a scalar"
        return Value(self.value ** other, children=[(other * self.value ** (other - 1), self)], op="**")

    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.value / other.value, children=[(1 / other.value, self), (-self.value / (other.value ** 2), other)], op="/")

    def __neg__(self):
        return Value(-self.value, children=[(-1, self)], op="neg")

    def log(self):
        # Añadir un pequeño valor epsilon para evitar el logaritmo de cero
        epsilon = 1e-15
        log_value = np.log(np.maximum(self.value, epsilon))
        grad = 1 / (self.value + epsilon)  # Añadir epsilon también aquí para evitar división por cero
        return Value(log_value, children=[(grad, self)], op="log")

    def relu(self):
        return Value(max(0, self.value), children=[(1 if self.value > 0 else 0, self)], op="relu")

    def sigmoid(self):
        # Perform the sigmoid function on the value
        value = 1 / (1 + np.exp(-self.value))

        # Directly calculate the derivative of sigmoid, which is sigmoid(x) * (1 - sigmoid(x))
        # The derivative should be in terms of the original value before applying sigmoid
        # We do not need to create a new Value object here, just use the numeric derivative
        derivative = value * (1 - value)

        # Return a new Value object representing the result of the sigmoid function
        # Its child is the original Value object (self) with the associated derivative
        return Value(value, children=[(derivative, self)], op="sigmoid")

    def backward(self):
        # Base case: if this is the root node
        if not self.children:
            self.grad = 1
            return
        # Recursive case: apply chain rule
        assert isinstance(self.grad, (int, float)), "self.grad must be a scalar"
        for coeff, child in self.children:
            assert isinstance(coeff, (int, float)), "coeff must be a scalar"
            child.grad += coeff * self.grad
            child.backward()

