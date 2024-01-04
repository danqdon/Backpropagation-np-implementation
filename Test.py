import numpy as np
from neural_network.DenseLayer import DenseLayer
from neural_network.Activation import Sigmoid
from neural_network.NeuralNetwork import NeuralNetwork
from neural_network.Activation import ReLU

def generate_xor_data():
    # XOR data inputs and outputs
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    return X, y

def main():
    # Generate training data
    X, y = generate_xor_data()

    # Create a neural network
    network = NeuralNetwork(DenseLayer(2, 4),
                            Sigmoid(),
                            DenseLayer(4, 1),
                            Sigmoid())

    # Train the network
    network.train(X, y, epochs=10000, learning_rate=0.1)

    # Make predictions
    predictions = network.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted: {predictions[i]}, True: {y[i]}")

if __name__ == "__main__":
    main()
