import numpy as np


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


def sigmoid_derivative(value):
    val = sigmoid(value)
    return val * (1 - val)


def relu(value):
    return max(0, value)


def relu_derivative(value):
    return 1 if value >= 0 else 0


def parametric_relu(value, slope_parameter):
    return max(value * slope_parameter, value)


def parametric_relu_derivative(value, slope_parameter):
    return 1 if value >= 0 else slope_parameter
