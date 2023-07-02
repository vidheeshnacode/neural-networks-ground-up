import numpy as np

# Inputs and Weights before

inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]] # batch of inputs

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# Transpose the weights for switching rows and columns so that it works with np.dot
output = np.dot(inputs, np.array(weights).T) + biases
# print(output)

# When weights2 and biases2 are defined, layer1 output will be input for layer2
# i.e., layer1_output = np.dot(inputs, np.array(weights).T) + biases
#       layer2_output = np.dot(layer1_outputs, np.array(weights2).T) + biases2


# Better way of doing this is using objects

np.random.seed(0)

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:

    # Weights and Biases are initialized
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# defining 2 hidden layers
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
# print(layer1.output)

layer2.forward(layer1.output)
print(layer2.output)