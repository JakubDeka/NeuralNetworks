import numpy as np
import random
from classes import NeuralNetwork

data = [[2, 3, 0.25, 0.5], [1, 0, -0.25, 0.25], [2, 6, 1, 0.5]]
data_or = [[1, 1, 1], [0, 1, 1], [1, 0, 1], [0, 0, 0]]
data_xor = [[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0]]
data_and = [[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]]

dataset = data_and

row = dataset[0]
x = row[:2]
y = row[2:]

snn = NeuralNetwork.SimpleNeuralNetwork(len(x), len(y))
snn.add_layer(6)
snn.add_layer(1)
snn.change_layer(0, 3)

print(snn)

# snn.calculate_network_output(x)
# snn.calculate_error(y)
# snn.calculate_gradients()
# snn.backpropagate()
# # print(snn)
# print(40*'-')
# print(snn)
# snn.apply_weight_changes(1)
# snn.calculate_network_output(x)
# snn.calculate_error(y)
# print(40*'-')
# print(snn)
# snn.calculate_gradients()
# print(20*"-"+"GRADIENT")
# print(40*'-')
# print(snn)
# snn.backpropagate()
# print(20*"-"+"BACKPROPAGATION")
# print(40*'-')
# print(snn)

for i in range(100000):
    row = random.choice(dataset)
    # row = dataset[0]
    x = row[:2]
    y = row[2:]
    snn.learn_single_iteration(x, y, 0.025)
    if (i + 1)%10000 == 0:
        print(snn)

print(snn.calculate_network_output(dataset[0][:2]))
print(snn.calculate_network_output(dataset[1][:2]))
print(snn.calculate_network_output(dataset[2][:2]))
print(snn.calculate_network_output(dataset[3][:2]))
