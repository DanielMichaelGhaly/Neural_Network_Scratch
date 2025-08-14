from layer import Layer
import numpy as np


class Dense(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        # Output is j x 1, Input is i x 1 so weights are i x j and bias is j x 1
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    # returns derivative of E with respect to Input X
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient # as it is same as bias_gradient
        return np.dot(self.weights.T, output_gradient)
        pass
