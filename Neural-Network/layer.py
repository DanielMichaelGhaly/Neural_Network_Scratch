class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # return output
    def forward(self, input):
        pass

    # update parameters and return input gradient
    def backward(self, output_gradient, learning_rate):
        pass