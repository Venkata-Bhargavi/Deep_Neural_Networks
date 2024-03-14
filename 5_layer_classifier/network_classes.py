import os
import cv2
import numpy as np

class Regularization:
    def __init__(self, l2_lambda=0.01):
        self.l2_lambda = l2_lambda

    def l2_regularization_loss(self, weights):
        return 0.5 * self.l2_lambda * np.sum(weights ** 2)

    def l2_regularization_gradient(self, weights):
        return self.l2_lambda * weights

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward_propagation(self, inputs):
        if self.dropout_rate > 0:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape) / (1 - self.dropout_rate)
            return inputs * self.mask
        else:
            return inputs

    def backward_propagation(self, gradient):
        if self.dropout_rate > 0:
            return gradient * self.mask
        else:
            return gradient

class Neuron:
    def __init__(self, input_dimensions):
        self.weights = None
        self.bias = None
        self.input_dimensions = input_dimensions
        self.aggregate_signal = None
        self.activation = None
        self.output = None
        self.layer = None

    def neuron(self, inputs):
        if self.weights is None or self.bias is None:
            raise ValueError("Weights and bias must be set before forward pass")
        if inputs.shape[-1] != self.input_dimensions:
            raise ValueError(f"Inputs shape ({inputs.shape}) is incompatible with neuron input dimensions ({self.input_dimensions})")
        self.aggregate_signal = np.sum(np.dot(inputs, self.weights) + self.bias)
        self.activation = self.layer.activation(self.aggregate_signal)
        self.output = self.activation

class Activation_Functions:
    def __init__(self, type):
        self.type = type

    def __call__(self, inputs):
        if self.type == "relu":
            return np.maximum(0, inputs)
        elif self.type == "sigmoid":
            return 1 / (1 + np.exp(-inputs))
        elif self.type == "tanh":
            return np.tanh(inputs)
        elif self.type == "softmax":
            exp_values = np.exp(inputs - np.max(inputs, axis=-1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        else:
            raise ValueError(f"Invalid activation function: {self.type}, Choose a valid activation function")

class Layer:
    def __init__(self, input_dimensions, output_dimensions, activation_type, normalization_type=None, regularization=None, dropout=None):
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.activation_type = activation_type
        self.activation = Activation_Functions(activation_type)
        self.neurons = [Neuron(input_dimensions) for _ in range(output_dimensions)]
        self.bias = np.random.randn()
        self.normalization_type = normalization_type
        self.regularization = regularization
        self.dropout = dropout

    def forward_propagation(self, inputs, targets, batch_size, training=True):
        num_samples = inputs.shape[0]
        print("num_samples: ",num_samples)
        num_batches = num_samples // batch_size

        total_loss = 0
        for batch_idx in range(num_batches):
            batch_inputs = inputs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_targets = targets[batch_idx * batch_size: (batch_idx + 1) * batch_size]

            if self.normalization_type == "z_score":
                # Apply Z-score normalization
                batch_inputs, mean, std_dev = z_score_normalization(batch_inputs)
            elif self.normalization_type == "min_max":
                # Apply Min-Max normalization
                batch_inputs, min_val, max_val = min_max_normalization(batch_inputs)

            outputs = []
            for neuron in self.neurons:
                neuron.weights = np.random.rand(self.input_dimensions)
                neuron.bias = self.bias
                neuron.layer = self
                if training:
                    batch_inputs = self.dropout.forward_propagation(batch_inputs)
                neuron.neuron(batch_inputs)
                outputs.append(neuron.output)
            # outputs = np.array(outputs)
            print("outputs: ",outputs)
            print("batch_targets: ",batch_targets)
            batch_loss = np.mean(np.square(outputs - batch_targets.astype(float)))  # Ensure batch_targets are converted to float
            if self.regularization:
                for neuron in self.neurons:
                    batch_loss += self.regularization.l2_regularization_loss(neuron.weights)
            total_loss += batch_loss

        # Average loss across all batches
        average_loss = total_loss / num_batches
        return outputs, average_loss

class MultiLayerPerceptron:
    def __init__(self, input_dimensions, hidden_layer_sizes, output_size, activation_types, normalization_type=None, regularization=None, dropout=None):
        self.input_dimensions = input_dimensions
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_size = output_size
        self.activation_types = activation_types
        self.normalization_type = normalization_type
        self.regularization = regularization
        self.dropout = dropout
        self.layers = []

        # Create input layer
        input_layer = Layer(input_dimensions, hidden_layer_sizes[0], activation_types[0], normalization_type, regularization, dropout)
        self.layers.append(input_layer)

        # Create hidden layers  (iterates till it creates total hidden layers requested in main method)
        for i in range(len(hidden_layer_sizes) - 1):
            hidden_layer = Layer(hidden_layer_sizes[i], hidden_layer_sizes[i + 1], activation_types[i + 1], normalization_type, regularization, dropout)
            self.layers.append(hidden_layer)

        # Create output layer
        output_layer = Layer(hidden_layer_sizes[-1], output_size, activation_types[-1], normalization_type, regularization, dropout)
        self.layers.append(output_layer)

    def forward_propagation(self, inputs, targets, batch_size, training=True):
        outputs = inputs
        for layer in self.layers:
            outputs, loss = layer.forward_propagation(outputs, targets, batch_size, training)
        return outputs, loss

def z_score_normalization(data):
    """
    Z-score normalization algorithm.

    Parameters:
    data (numpy array): Input data to be normalized.

    Returns:
    numpy array: Normalized data.
    """
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data, mean, std_dev

def min_max_normalization(data):
    """
    Min-Max normalization algorithm.

    Parameters:
    data (numpy array): Input data to be normalized.

    Returns:
    numpy array: Normalized data.
    """
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data, min_val, max_val

def load_images_from_directory(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            # Assuming format is label_filename.jpg
            label = filename.split("_")[0]  # Extract label from filename
            try:
                if label == "Au":
                    images.append(cv2.imread(img_path))
                    labels.append(0)
                if label == "Sp":
                    images.append(cv2.imread(img_path))
                    labels.append(1)


            except ValueError:
                print(f"Ignoring file {filename}: Invalid label '{label}'")
                continue
    return images, labels
