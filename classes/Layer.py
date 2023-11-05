import numpy as np
from classes import Neuron as n


class Layer:

    def __init__(self, input_length, number_of_neurons, activation_function="sigmoid", activation_function_parameter=1):
        self.input = None
        self.input_size = input_length
        self.neurons = np.array([n.Neuron(self.input_size) for i in range(number_of_neurons)])
        self.no_neurons = number_of_neurons
        self.activation_function = activation_function
        self.activation_function_parameter = activation_function_parameter
        self.gradient = np.zeros(self.no_neurons)

    def __str__(self):
        neurons_info = "\n".join([str(i) + " " + neuron.__str__() for i, neuron in enumerate(self.neurons)])
        return f"Number of neurons in layer: {self.no_neurons}\n" \
               f"Layer input: {self.input}\n" \
               f"Neurons:\n{neurons_info}\n" \
               f"Layer activation function: {self.activation_function}\n" \
               f"Layer gradient:\n{self.gradient}\n"

    def change_input_size(self, new_input_size):
        self.input_size = new_input_size
        self.change_number_of_neurons(self.no_neurons)

    def change_number_of_neurons(self, new_number_of_neurons):
        self.neurons = np.array([n.Neuron(self.input_size) for i in range(new_number_of_neurons)])
        self.no_neurons = new_number_of_neurons
        self.gradient = np.zeros(self.no_neurons)

    def calculate_layer_output(self, bias):
        layer_output = np.array([])
        for neuron in self.neurons:
            layer_output = np.hstack((layer_output, neuron.calculate_neuron_output(self.input, self.activation_function,
                                                                                   self.activation_function_parameter,
                                                                                   bias)))
        return layer_output

    def calculate_layers_gradient(self):
        for i, neuron in enumerate(self.neurons):
            self.gradient[i] = neuron.calculate_neuron_derivative(self.activation_function,
                                                                  self.activation_function_parameter)

# layer = Layer(len([2, 1]), 2)
# layer.input = [2, 1]
# print(layer)
# print(layer.calculate_layer_output(1))