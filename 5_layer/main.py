import os
import shutil
import numpy as np
from network_classes import Regularization, Dropout, MultiLayerPerceptron, load_images_from_directory
from sklearn.model_selection import train_test_split
import cv2

def get_image_dimensions(image_path):
    sample_image = cv2.imread(image_path)
    height, width, channels = sample_image.shape
    return height, width, channels

# Path to the directory containing the images
authentic_dir = "/Users/bhargavi/PycharmProjects/Deep_Neural_Networks/5_layer/images/authentic"
fake_dir = "/Users/bhargavi/PycharmProjects/Deep_Neural_Networks/5_layer/images/fake_images"

# List all the image file paths in the authentic directory
authentic_images = [os.path.join(authentic_dir, file) for file in os.listdir(authentic_dir) if file.endswith(".jpg") or file.endswith(".png")]

# List all the image file paths in the fake_images directory
fake_images = [os.path.join(fake_dir, file) for file in os.listdir(fake_dir) if file.endswith(".jpg") or file.endswith(".png")]

# Combine the lists of image file paths
images = authentic_images + fake_images

# Assuming you have labels corresponding to authentic and fake images
authentic_labels = [0] * len(authentic_images)  # 0 represents authentic
fake_labels = [1] * len(fake_images)  # 1 represents fake
labels = authentic_labels + fake_labels
print("Number of images:", len(images))
print("Number of labels:", len(labels))


# Get dimensions of a sample image
sample_image_path = images[0]
height, width, channels = get_image_dimensions(sample_image_path)
# Calculate input dimensions
input_dimensions = height * width * channels
print(height,width,channels)
print("Input dimensions:", input_dimensions)


# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
print("Number of training images:", len(train_images))
print("Number of training labels:", len(train_labels))
print("Number of testing images:", len(test_images))
print("Number of testing labels:", len(test_labels))




# Create directories for train and test images
train_dir = '/Users/bhargavi/PycharmProjects/Deep_Neural_Networks/5_layer/images/train_images'
test_dir = '/Users/bhargavi/PycharmProjects/Deep_Neural_Networks/5_layer/images/test_images'

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
# Copy train images to train directory
for image_path in train_images:
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(train_dir, image_name))
# Copy test images to test directory
for image_path in test_images:
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(test_dir, image_name))



def main():
    # Example usage:
    # input_dimensions = 224 * 224 * 3  # Update input dimensions based on image size
    hidden_layer_sizes = [10, 8, 8, 4]  # Define layer sizes according to the specification
    output_size = 1
    activation_types = ["relu", "relu", "relu", "relu", "sigmoid"]  # Activation functions for each layer

    # Choose normalization type
    normalization_type = "min_max"  # Choose either "z_score" or "min_max"

    # Initialize regularization and dropout instances
    regularization = Regularization(l2_lambda=0.01)
    dropout = Dropout(dropout_rate=0.2)

    # Create MultiLayerPerceptron with normalization, regularization, and dropout
    mlp = MultiLayerPerceptron(input_dimensions, hidden_layer_sizes, output_size, activation_types, normalization_type, regularization, dropout)

    # Load images from directory
    train_images, train_labels = load_images_from_directory("train_directory_path")
    test_images, test_labels = load_images_from_directory("test_directory_path")

    # Convert images and labels to numpy arrays
    train_images = np.array(train_images)
    test_images = np.array(test_images)
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # Reshape images to match input dimensions
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Set batch size
    batch_size = 10

    # Perform forward propagation with normalization, regularization, dropout, and mini-batch approach
    outputs, loss = mlp.forward_propagation(train_images, train_labels, batch_size)
    print("Training Loss:", loss)

    # Test the model
    test_outputs, test_loss = mlp.forward_propagation(test_images, test_labels, batch_size, training=False)
    print("Test Loss:", test_loss)

# if __name__ == "__main__":
#     main()
