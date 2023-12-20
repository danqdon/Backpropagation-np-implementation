from Value import Value
from nn.NeuralNetwork import NeuralNetwork
from nn.FullyConnectedLayer import FullyConnected
from nn.Activations import Sigmoid, ReLU


# Mock data for testing: Simple XOR problem
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_outputs = [[0], [1], [1], [0]]

# Initialize the neural network
nn = NeuralNetwork(
    FullyConnected(2, 5),
    ReLU(),
    FullyConnected(5, 3),
    Sigmoid(),
    FullyConnected(3, 1),
    Sigmoid()
)

# Train the neural network
nn.train(input_data, target_outputs, alpha=0.1, epoch=100)

# Print the weights and gradients after training
print("Weights and Gradients after training:")
for layer in nn.layers:
    if isinstance(layer, FullyConnected):
        for i in range(len(layer.w)):
            for j in range(len(layer.w[i])):
                print(f"Weight: {layer.w[i][j].value}, Gradient: {layer.w[i][j].grad}")

# Test predictions
predictions = nn.predict(input_data)
for i in range(len(input_data)):
    prediction_value = predictions[i][0].value if isinstance(predictions[i][0], Value) else predictions[i][0]
    print(f"Input: {input_data[i]}, Predicted: {prediction_value}")
