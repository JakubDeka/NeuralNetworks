import numpy as np
import random
from classes import functions as fs


class Neuron:

    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = np.array([random.random()-0.5 for i in range(input_size)])
        self.weights_change = np.zeros(self.input_size)
        self.rest_change = np.zeros(self.input_size)
        self.output = 0

    def __str__(self):
        return f"Neuron weights:\n{self.weights}\n" \
               f"Weight changes:\n{self.weights_change}"

    def calculate_neuron_output(self, input, activation_function, activation_function_parameter: float, bias):
        value = np.dot(self.weights, input) + bias
        if activation_function == "linear":
            pass
        elif activation_function == "sigmoid":
            value = fs.sigmoid(value)
        elif activation_function == "relu":
            value = fs.relu(value)
        elif activation_function == "parametric_relu":
            value = fs.parametric_relu(value, activation_function_parameter)
        self.output = value
        return value

    def calculate_neuron_derivative(self, activation_function, activation_function_parameter: float):
        if activation_function == "linear":
            return 1
        elif activation_function == "sigmoid":
            return fs.sigmoid_derivative(self.output)
        elif activation_function == "relu":
            return fs.relu_derivative(self.output)
        elif activation_function == "parametric_relu":
            return fs.parametric_relu_derivative(self.output, activation_function_parameter)

    def calculate_delta(self, input, rest_of_derivative):
        # derivative in respect to the appropraite weight
        for i, weight in enumerate(self.weights):
            self.weights_change[i] = input[i] * rest_of_derivative
            self.rest_change[i] = self.weights[i] * rest_of_derivative
        # return self.weights_change

# neuron = Neuron(2)
# print(neuron)
# print(f"value: {neuron.calculate_neuron_output([2, 1], 'sigmoid', 1)}")
# print(f"derivative: {neuron.calculate_neuron_derivative('sigmoid')}")
# print(neuron)
