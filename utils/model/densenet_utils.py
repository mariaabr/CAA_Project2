import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet169

def build_densenet169_model(input_shape=(28, 28, 1), num_classes=2, dropout_rate=0.5):
    """Build a DenseNet-169 model adapted for grayscale medical images."""
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Convert grayscale to 3-channel (DenseNet expects RGB input)
    x = layers.Conv2D(3, (1, 1), padding='same')(inputs)
    
    # Resize to 224x224 as expected by DenseNet
    x = layers.Resizing(224, 224)(x)
    
    # Load pre-trained DenseNet-169 without top layer
    base_model = DenseNet169(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Connect the base model
    x = base_model(x)
    
    # Add classification head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='fc')(x)
    
    # Create the full model
    model = models.Model(inputs, outputs)
    
    return model