from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from classes import NeuralNetwork
import numpy as np
import cv2
import os


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

# Load images from the directory
images_directory = "/Users/bhargavi/PycharmProjects/Deep_Neural_Networks/Network_Training/images"
X, y = load_images(images_directory)

# Define your parameter grid
param_grid = {
    'learning_rate': [0.1, 0.01, 0.001],
    'num_epochs': [50, 100, 200],
    'batch_size': [32, 64, 128],
    'hidden_layer_sizes': [(64, 32), (128, 64), (256, 128)],
    'activation': ['relu', 'tanh'],
    'weight_init': ['xavier', 'he']
}

# Create an instance of your neural network
model = NeuralNetwork()

# Split your data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize GridSearchCV with your model, parameter grid, and scoring metric
grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=3)

try:
    # Fit the grid search to your training data
    grid_search.fit(X_train, y_train)

    # Get the best parameters and their corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Train the model with the best parameters on the entire training set
    best_model = grid_search.best_estimator_
    best_model.fit(X_train, y_train)

    # Evaluate the best model on the validation set
    y_pred = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)

    print("Best Parameters:", best_params)
    print("Best Score (Negative MSE):", best_score)
    print("Mean Squared Error on Validation Set:", mse)

except KeyboardInterrupt:
    print("Training interrupted by keyboard.")
