# Assuming Value, NeuralNetwork, and mse_loss are defined correctly
from Value import Value
from NeuralNetworkOld import NeuralNetwork
from nn.NeuralNetwork import NeuralNetwork
from nn.FullyConnectedLayer import FullyConnected
from Utils import mse_loss

# Mock data for testing: Simple XOR problem
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_outputs = [[0], [1], [1], [0]]

# Initialize the neural network
c1, c2 = FullyConnected(input_size=2, output_size=4), FullyConnected(input_size=4, output_size=1)
nn = NeuralNetwork(
        c1,
    c2
    )

# Train the neural network
nn.train(input_data, target_outputs, alpha=0.1, epoch=100)

# Print the weights and gradients after training
print("Weights and Gradients after training:")
for i in range(len(c1.w)):
    for j in range(len(c1.w[i])):
        print(f"w1[{i}][{j}].value: {c1.w[i][j].value}, w1[{i}][{j}].grad: {c1.w[i][j].grad}")

for i in range(len(c2.w)):
    for j in range(len(c2.w[i])):
        print(f"w2[{i}][{j}].value: {c2.w[i][j].value}, w2[{i}][{j}].grad: {c2.w[i][j].grad}")

predictions = nn.predict(input_data)
print(predictions[0][0].value)
for i in range(len(input_data)):
    print(f"Input: {input_data[i]}, Predicted: {predictions[i][0].value}")