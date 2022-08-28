import numpy as np
import time
np.set_printoptions(precision=2)

# Needs more attention
# X = np.load('data-files/X.npy')
# y = np.array(([90, 0, 180, 0, 0, 0, 90, 0, 0]), dtype=float)

X = np.array(([2, 9, 4.5, 1.2], [1, 5, 2.3, 2.4], [3, 6, 1.5, 5]), dtype=float)
y = np.array(([9, 3, 3], [8, 6, 5], [8, 9, 7]), dtype=float)

# scale units
X = X/np.amax(X, axis=0)  # maximum of X array
y = y/np.amax(y, axis=0)  # max test score is 100

class NeuralNetwork(object):
    def __init__(self):
        # parameters
        self.inputSize = X.shape[-1]
        self.outputSize = y.shape[-1]
        self.hiddenSize = 3

        # weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) weight matrix from input to hidden layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) weight matrix from hidden to output layer

    def forward(self, X):
        # forward propagation through our network
        self.z = np.dot(X, self.W1)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z3 = np.dot(self.z2, self.W2)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def sigmoid(self, s):
        # activation function
        return 1/(1+np.exp(-s))

    def sigmoid_prime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_error = y - o  # error in output
        self.o_delta = self.o_error*self.sigmoid_prime(o)  # applying derivative of sigmoid to error

        self.z2_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = self.z2_error*self.sigmoid_prime(self.z2)  # applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta)  # adjusting first set (input --> hidden) weights
        self.W2 += self.z2.T.dot(self.o_delta)  # adjusting second set (hidden --> output) weights

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)

NN = NeuralNetwork()
for i in range(1000):  # trains the NN 1,000 times
    print("Input:")
    print(X)
    print("Output:")
    print(y)
    print("Predicted Output:")
    print(NN.forward(X))
    print("Loss:")

    print('{:.6f}'.format(np.mean(np.square(y - NN.forward(X)))))
    #print('{:.6f}'.format(np.mean(np.square(y.shape - NN.forward(X).shape))))
    NN.train(X, y)
    time.sleep(.1)
