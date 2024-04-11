import numpy as np
class NeuralNetwork:
    def __init__(self, learning_rate=0.001, num_epochs=100, batch_size=32, hidden_layer_sizes=(64, 32), activation='relu', weight_init='xavier'):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.weight_init = weight_init
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

    def fit(self, X_train, y_train, max_epochs=100, min_loss=0.01):
        '''
        :param X_train: training data
        :param y_train: target of training data
        :param max_epochs: maximum number of epochs to train
        :param min_loss: minimum loss threshold to stop training
        '''
        for epoch in range(max_epochs):
            total_loss = 0
            for X, y in zip(X_train, y_train):
                self.backward_propagation(X, y, self.learning_rate)
                total_loss += np.mean(np.square(self.forward_propagation(X) - y))
            avg_loss = total_loss / len(X_train)
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {avg_loss:.4f}")
            if avg_loss < min_loss:
                print(f"Reached minimum loss threshold of {min_loss}. Stopping training.")
                break

    def predict(self, X):
        '''
        Predict using the trained neural network
        :param X: input data
        :return: predicted values
        '''
        predictions = []
        for x in X:
            predictions.append(self.forward_propagation(x))
        return np.array(predictions)

    def get_params(self, deep=True):
        '''
        Get parameters for the neural network
        '''
        return {
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'activation': self.activation,
            'weight_init': self.weight_init
        }

    def set_params(self, **parameters):
        '''
        Set parameters for the neural network
        '''
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
