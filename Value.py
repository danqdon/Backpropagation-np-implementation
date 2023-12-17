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
        for coeff, child in self.children:
            child.grad += coeff * self.grad
            child.backward()
