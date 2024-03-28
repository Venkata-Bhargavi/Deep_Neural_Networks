from classes import NeuralNetwork, DenseLayer, relu, sigmoid
import numpy as np
import os
import cv2

def load_images(directory):
    '''
    :param directory: Directory of entire image data
    :return: images and its corresponding labels
    '''
    images = []
    labels = []
    for label, folder_name in enumerate(os.listdir(directory)):
        folder_path = os.path.join(directory, folder_name)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (100, 100))
            images.append(image.flatten() / 255.0)  # Normalize pixel values
            labels.append(label)
    return np.array(images), np.array(labels)

def main():
    # Load and preprocess images
    images_directory = "/Users/bhargavi/PycharmProjects/Deep_Neural_Networks/Network_Training/images"
    X, y = load_images(images_directory)
    num_classes = len(np.unique(y))
    y_one_hot = np.eye(num_classes)[y]

    # Split data into training and testing sets (80-20 split)
    num_samples = X.shape[0]
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_one_hot[:split_idx], y_one_hot[split_idx:]


    model = NeuralNetwork()

    # Add layers to the model
    model.add_layer(DenseLayer(input_size=X.shape[1], output_size=10, activation=relu))
    model.add_layer(DenseLayer(input_size=10, output_size=8, activation=relu))
    model.add_layer(DenseLayer(input_size=8, output_size=8, activation=relu))
    model.add_layer(DenseLayer(input_size=8, output_size=4, activation=relu))
    model.add_layer(DenseLayer(input_size=4, output_size=num_classes, activation=sigmoid))

    # Train the neural network
    epochs = 100
    learning_rate = 0.001
    for epoch in range(epochs):
        # Iterating over each epoch
        total_loss = 0
        for X, y in zip(X_train, y_train):
            X = X.reshape(1, -1)  # Reshape X to (1, num_features) for single sample
            y = y.reshape(1, -1)  # Reshape y to (1, num_classes) for single sample
            model.backward_propagation(X, y, learning_rate)
            total_loss += np.mean(np.square(model.forward_propagation(X) - y))
        avg_loss = total_loss / len(X_train)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluate the model on the test set
    test_loss = 0
    for X, y in zip(X_test, y_test):
        X = X.reshape(1, -1)  # Reshape X to (1, num_features) for single sample
        y = y.reshape(1, -1)  # Reshape y to (1, num_classes) for single sample
        prediction = model.forward_propagation(X)
        test_loss += np.mean(np.square(prediction - y))
    avg_test_loss = test_loss / len(X_test)
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    main()
