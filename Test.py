from Value import Value
from NeuralNetwork import NeuralNetwork

def main():
    # XOR problem data
    input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    target_outputs = [[0], [1], [1], [0]]

    # Initialize a larger neural network
    nn = NeuralNetwork(2, [10, 10, 10], 1, seed=42)  # Three hidden layers with 10 neurons each

    # Training the network
    nn.train(input_data, target_outputs, alpha=0.1, epoch=100)

    # Testing the network
    test_predictions = nn.predict([[0, 0], [0, 1], [1, 0], [1, 1]])

    # Display the predictions
    for inp, pred in zip(input_data, test_predictions):
        print(f"Input: {inp}, Predicted: {pred}")

if __name__ == "__main__":
    main()
