import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
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

def create_augmentation_pipeline():
    """Create a data augmentation pipeline for chest X-ray images."""
    data_augmentation = tf.keras.Sequential([
        # Randomly rotate images by up to 15 degrees
        layers.RandomRotation(0.08),  # 0.08 ~= 15 degrees in radians
        # Randomly shift images horizontally and vertically
        layers.RandomTranslation(0.1, 0.1),
        # Randomly zoom in or out
        layers.RandomZoom(0.1),
        # Adjust contrast
        layers.RandomContrast(0.1),
    ])
    return data_augmentation

def apply_augmentation(images, labels, batch_size=32, seed=42):
    """Apply data augmentation to images and labels.
    
    Args:
        images: Preprocessed images (normalized, reshaped)
        labels: One-hot encoded labels
        batch_size: Batch size for the dataset
        seed: Random seed for reproducibility
        
    Returns:
        tf.data.Dataset: A dataset with augmented images
    """
    # Create tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Create augmentation pipeline
    augmentation = create_augmentation_pipeline()
    
    # Define the preprocessing function
    def augment(image, label):
        # Apply data augmentation with fixed seed
        image = augmentation(image, training=True)
        return image, label
    
    # Set seed for shuffling
    tf.random.set_seed(seed)
    
    # Apply augmentation and shuffling with seed
    augmented_dataset = dataset.shuffle(buffer_size=len(images), seed=seed)
    augmented_dataset = augmented_dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    augmented_dataset = augmented_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return augmented_dataset

def load_and_preprocess_data(data_path="../data/", augment=False, batch_size=32, seed=42):
    """Load and preprocess data, with optional augmentation.
    
    Args:
        data_path: Path to the data directory
        augment: Whether to apply data augmentation to training data
        batch_size: Batch size if using augmentation
        seed: Random seed for reproducibility
        
    Returns:
        If augment=False: Regular numpy arrays (backward compatible)
        If augment=True: tf.data.Dataset for training, numpy arrays for test/val
    """
    # Set NumPy random seed for reproducibility
    np.random.seed(seed)
    
    # Set TensorFlow random seed for reproducibility
    tf.random.set_seed(seed)
    
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
    
    if augment:
        print("\nApplying data augmentation to training data...")
        train_dataset = apply_augmentation(train_images, train_labels, batch_size, seed=seed)
        return train_dataset, (test_images, test_labels), (val_images, val_labels)
    else:
        return train_images, train_labels, test_images, test_labels, val_images, val_labels