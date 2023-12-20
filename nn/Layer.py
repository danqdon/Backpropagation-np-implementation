from abc import ABC, abstractmethod
class Layer(ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backprop(self, grad):
        pass

    @abstractmethod
    def update_weights(self, alpha):
        pass
