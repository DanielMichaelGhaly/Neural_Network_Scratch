# I made it with 3 layers Neural Network
# 2 nodes for input (X) and 3 nodes for processing and 1 node for output Y

from dense_layer import Dense
from hyperbolic_tangent_activation import Tanh
from mean_squared_error import mse, mse_prime
import numpy as np

X = np.reshape([[0,0], [0,1], [1,0], [1,1]], (4,2,1))
Y = np.reshape([[0], [1], [1], [0]], (4,1,1))

# Activation after each hidden layer
network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

epochs = 10000
learning_rate = 0.1

# train
for e in range(epochs):
    error = 0
    for x, y in zip(X, Y):
        # Forward
        output = x
        for layer in network:
            output = layer.forward(output)

        # error
        error += mse(y, output)

        # backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad - layer.backward(grad, learning_rate)

    error /= len(X)

    print('%d/%d, error=%f' % (e + 1, epochs, error))