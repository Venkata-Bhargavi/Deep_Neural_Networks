import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        '''
        adds layers as requested
        '''
        self.layers.append(layer)

    def forward_propagation(self, X):
        '''
        Forward propogation
        :param X: training data
        :param layers: all the defined layers
        '''
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward_propagation(self, X, y, learning_rate):
        '''
        :param error: error from forward prop
        :param layers: all the layers
        :param learning_rate: hyperparameter
        :return: error
        '''
        output = self.forward_propagation(X)
        error = output - y
        for layer in reversed(self.layers):
            error = layer.backward(error, learning_rate)

    def train(self, X_train, y_train, epochs, learning_rate):
        '''
        :param X_train: training data
        :param y_train: target of training data
        :param epochs:  no of epoch to iterate
        :param learning_rate: hyperparameter
        '''
        for epoch in range(epochs):
            total_loss = 0
            for X, y in zip(X_train, y_train):
                self.backward_propagation(X, y, learning_rate)
                total_loss += np.mean(np.square(self.forward_propagation(X) - y))
            avg_loss = total_loss / len(X_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        self.activation = activation
        self.input = None
        self.output = None

    def forward(self, X):
        self.input = X
        self.output = self.activation(np.dot(X, self.weights) + self.bias)
        return self.output

    def backward(self, error, learning_rate):
        error *= self.activation(self.output, derivative=True)
        delta_weights = np.dot(self.input.T, error)
        self.weights -= learning_rate * delta_weights
        self.bias -= learning_rate * np.sum(error, axis=0, keepdims=True)
        return np.dot(error, self.weights.T)

def relu(x, derivative=False):
    '''
    implements logic of relu activation function
    '''
    if derivative:
        return np.where(x <= 0, 0, 1)
    return np.maximum(0, x)

def sigmoid(x, derivative=False):
    '''
    implements logic of sigmoid activation function to use in final layer for classification
    '''
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))
