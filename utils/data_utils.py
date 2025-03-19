import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data(data_path="../data/"):
    train_images = np.load(data_path + "train_images.npy")
    train_labels = np.load(data_path + "train_labels.npy")
    test_images = np.load(data_path + "test_images.npy")
    test_labels = np.load(data_path + "test_labels.npy")
    val_images = np.load(data_path + "val_images.npy")
    val_labels = np.load(data_path + "val_labels.npy")
    print("\nDataset shapes - Original:")
    print("Train - images:", train_images.shape, " labels:", train_labels.shape)
    print("Val   - images:", val_images.shape, " labels:", val_labels.shape)
    print("Test  - images:", test_images.shape, " labels:", test_labels.shape)
    return train_images, train_labels, test_images, test_labels, val_images, val_labels

def preprocess_data(images, labels):
    # Reshape to include channel dimension (grayscale = 1 channel)
    images = images.reshape(images.shape[0], 28, 28, 1)
    
    # Normalize pixel values
    images = images.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    labels = to_categorical(labels, 2)
    
    return images, labels

def load_and_preprocess_data(data_path="../data/"):
    train_images, train_labels, test_images, test_labels, val_images, val_labels = load_data(data_path)
    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)
    val_images, val_labels = preprocess_data(val_images, val_labels)

    print("\nDataset shapes - After preprocessing:")
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Val images shape:", val_images.shape)
    print("Val labels shape:", val_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
    return train_images, train_labels, test_images, test_labels, val_images, val_labels