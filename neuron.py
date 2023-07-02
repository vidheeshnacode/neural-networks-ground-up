import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# ideally the optimizer is going to tune the values of weights and bias

layer_outputs = []  # Output of the current layer

for neuron_weight, neuron_bias in zip(weights, bias):
    neuron_output = 0  # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weight):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

# # Understanding dot product

# inputs = [1,2,3,2.5]
# weights = [0.2,0.8,-0.5,1.0]
# bias = 2
#
# output = np.dot(weights, inputs) + bias
# print(output) # 4.8

# dot product of a layer of neurons

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)

# np.dot(weights, inputs) is equivalent to
# [np.dot(weight[0], inputs), np.dot(weights[1], inputs), np.dot(weights[2], inputs)]


