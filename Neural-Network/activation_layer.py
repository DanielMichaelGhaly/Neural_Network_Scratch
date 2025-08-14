from layer import Layer
import numpy as np

class Activation(Layer):
    # activation prime -> derivative of activation function with respect to x
    # used in backpropagation to compute how much each neuron should adjust its weights.
    # z = Wx + b (z is last output before the activation function)
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    #Update parameters and return input gradient
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))