# Assuming Value, NeuralNetwork, and mse_loss are defined correctly
from Value import Value
from NeuralNetwork import NeuralNetwork
from Utils import mse_loss

# Mock data for testing: Simple XOR problem
input_data = [[0, 0], [0, 1], [1, 0], [1, 1]]
target_outputs = [[0], [1], [1], [0]]

# Initialize the neural network
nn = NeuralNetwork(2, 2, 1, seed=42)

# Train the neural network
nn.train(input_data, target_outputs, alpha=0.1, epoch=100)

# Print the weights and gradients after training
print("Weights and Gradients after training:")
for i in range(len(nn.w1)):
    for j in range(len(nn.w1[i])):
        print(f"w1[{i}][{j}].value: {nn.w1[i][j].value}, w1[{i}][{j}].grad: {nn.w1[i][j].grad}")

for i in range(len(nn.w2)):
    for j in range(len(nn.w2[i])):
        print(f"w2[{i}][{j}].value: {nn.w2[i][j].value}, w2[{i}][{j}].grad: {nn.w2[i][j].grad}")

predictions = nn.predict(input_data)
print(predictions[0][0].value)
for i in range(len(input_data)):
    print(f"Input: {input_data[i]}, Predicted: {predictions[i][0].value}")