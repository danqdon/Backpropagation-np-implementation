import numpy as np
from neural_network.DenseLayer import DenseLayer
from neural_network.SigmoidActivation import SigmoidActivation
from neural_network.NeuralNetwork import NeuralNetwork
from neural_network.ReluActivation import ReluActivation

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
    network = NeuralNetwork()
    network.add_layer(DenseLayer(2, 3))
    network.add_layer(ReluActivation())
    network.add_layer(DenseLayer(3, 1))
    network.add_layer(SigmoidActivation())

    # Train the network
    network.train(X, y, epochs=10000, learning_rate=0.1)

    # Make predictions
    predictions = network.predict(X)
    for i in range(len(X)):
        print(f"Input: {X[i]}, Predicted: {predictions[i]}, True: {y[i]}")

if __name__ == "__main__":
    main()
