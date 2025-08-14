# I made it with 3 layers Neural Network
# 2 nodes for input (X) and 3 nodes for processing and 1 node for output Y

from dense_layer import Dense
from hyperbolic_tangent_activation import Tanh
from mean_squared_error import mse, mse_prime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def forward_once(network, x_colvec):
    out = x_colvec
    for layer in network:
        out = layer.forward(out)
    return out

def visualize_xor_3d(network, steps=80, bounds=(0.0,1.0,0.0,1.0), show_points=True):
    x_min, x_max, y_min, y_max = bounds
    xs = np.linspace(x_min, x_max, steps)
    ys = np.linspace(y_min, y_max, steps)
    XX, YY = np.meshgrid(xs, ys)

    ZZ_raw = np.zeros_like(XX, dtype=float)
    for i in range(steps):
        for j in range(steps):
            x1, x2 = XX[i, j],  YY[i, j]
            z = forward_once(network, np.array([[x1],[x2]]))
            ZZ_raw[i, j] = float(z)

        ZZ_prob = (ZZ_raw + 1.0) / 2.0

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(XX, YY, ZZ_prob, rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.85)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('output (mapped to 0..1)')
        ax.set_title('XOR decision surface (tanh output mapped to probability)')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(0.0, 1.0)

        if show_points:
            X_pts = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
            Y_pts = np.array([0, 1, 1, 0], dtype=float)
            Z_pts = []
            for p in X_pts:
                z = forward_once(network, p.reshape(2, 1))
                Z_pts.append(float((z + 1) / 2))  # map tanh to 0..1
            Z_pts = np.array(Z_pts)
            ax.scatter(X_pts[:, 0], X_pts[:, 1], Z_pts, s=60)
            ax.scatter(X_pts[:, 0], X_pts[:, 1], Y_pts, s=80, marker='^')

        plt.tight_layout()
        plt.show()



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
            grad = layer.backward(grad, learning_rate)

    error /= len(X)

    print('%d/%d, error=%f' % (e + 1, epochs, error))
visualize_xor_3d(network, steps = 120)