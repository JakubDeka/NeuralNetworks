import copy
import random

import numpy as np
from classes import Layer as l
from classes import nn_auxiliary as aux


class SimpleNeuralNetwork:

    def __init__(self, input_length, output_length):
        self.bias = 1
        self.output = None
        self.target = None
        self.error = None
        self.layers = np.array([l.Layer(input_length, output_length)])
        self.no_layers = 1
        self.pred_error_derivative = None
        self.current_layer_derivatives = None

    def __str__(self):
        layers_info = "\n".join(["Layer " + str(i) + "\n" + layer.__str__() for i, layer in enumerate(self.layers)])
        if self.error is not None:
            mean_error = f"mean error: {np.mean(self.error)}"
        else:
            mean_error = ""
        return f"{40 * '-'}" \
               f"\nSNN input size: {self.layers[0].input_size}\n" \
               f"Number of layers: {self.no_layers}\n\n" \
               f"Layers:\n{layers_info}\n" \
               f"Current error derivative: {self.pred_error_derivative}\n" \
               f"Current error: {self.error}\n"+mean_error

    def add_layer(self, number_of_neurons, activation_function="sigmoid"):
        previous_layer = self.layers[-1]
        # print(f"previous_layer:\n{previous_layer}")
        self.layers = np.hstack((self.layers, l.Layer(previous_layer.no_neurons, number_of_neurons,
                                                      activation_function)))
        self.no_layers += 1

    def change_layer(self, layer_number: int, number_of_neurons: int = 0, activation_function: str =""):
        assert layer_number >= 0 and layer_number < self.no_layers, "There is no such layer available"
        if number_of_neurons > 0:
            if layer_number == self.no_layers - 1:
                print("Cannot change number of neurons in output layer")
            self.layers[layer_number].change_number_of_neurons(number_of_neurons)
            self.layers[layer_number + 1].change_input_size(number_of_neurons)
        if len(activation_function) > 0:
            aux.assert_correct_activation_function(activation_function)
            self.layers[layer_number].activation_function = activation_function

    def calculate_network_output(self, input):
        current_layer_output = np.array([])
        current_input = input
        for layer in self.layers:
            layer.input = current_input
            current_layer_output = layer.calculate_layer_output(self.bias)
            current_input = current_layer_output
        self.output = current_layer_output
        return current_layer_output

    def predict(self, input):
        current_layer_output = np.array([])
        current_input = input
        for layer in self.layers:
            layer.input = current_input
            current_layer_output = layer.calculate_layer_output(self.bias)
            current_input = current_layer_output
        self.output = current_layer_output
        return current_layer_output

    def calculate_error(self, target_value):
        self.target = target_value
        self.pred_error_derivative = self.output - self.target
        self.error = np.square(self.pred_error_derivative) / 2
        return self.error

    def calculate_gradients(self):
        for layer in self.layers:
            layer.calculate_layers_gradient()

    def backpropagate(self):
        current_layer_number = 1 # from the back of the list
        current_layer_index_in_list = self.no_layers - current_layer_number # we start with 0 and end with no_layers-1
        current_layer = self.layers[current_layer_index_in_list]
        current_input = current_layer.input
        # initialize derivatives as error derivative * last_layer's gradient
        rest_of_derivatives = np.array(
            [self.pred_error_derivative[i] * current_layer.gradient[i] for i in range(len(self.pred_error_derivative))])
        while current_layer_index_in_list >= 0:
            current_layer = self.layers[current_layer_index_in_list]
            current_input = current_layer.input
            next_derivatives = np.zeros(self.layers[current_layer_index_in_list - 1].no_neurons)
            # calculate each neurons changes, as well as derivatives used in next layers
            for i, neuron in enumerate(current_layer.neurons):
                # print(current_input)
                # print(rest_of_derivatives[i])
                neuron.calculate_delta(current_input, rest_of_derivatives[i])
                if not current_layer_index_in_list == 0:
                    next_derivatives = np.add(next_derivatives, neuron.rest_change)
                # print(neuron.weights_change)
                # print(neuron.rest_change)
            current_layer_index_in_list -= 1
            rest_of_derivatives = next_derivatives

    def apply_weight_changes(self, learning_rate):
        for layer in self.layers:
            for neuron in layer.neurons:
                neuron.weights -= learning_rate * neuron.weights_change
                for i in range(neuron.input_size):
                    if neuron.weights[i] > 10:
                        neuron.weights[i] = 10;
                    elif neuron.weights[i] < -10:
                        neuron.weights[i] = -10;
                neuron.weights_change = np.zeros(neuron.input_size)
            layer.gradient = np.zeros(layer.no_neurons)

    def learn_single_iteration(self, input, target, learning_rate=0.05):
        self.calculate_network_output(input)
        self.calculate_error(target)
        self.calculate_gradients()
        self.backpropagate()
        self.apply_weight_changes(learning_rate)
