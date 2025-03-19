from tensorflow.keras import layers, models, callbacks

def build_alexnet(input_shape=(28, 28, 1), num_classes=2, dropout_rate=0.5):
    """
    Build an AlexNet model adapted for small input size (28x28)
    """
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(96, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Second convolutional block
    x = layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Third convolutional block
    x = layers.Conv2D(384, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Fourth convolutional block
    x = layers.Conv2D(384, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Fifth convolutional block
    x = layers.Conv2D(256, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=2, strides=2)(x)
    
    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(4096)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    
    x = layers.Dense(4096)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)  # Named for the dropout scheduler
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

class DropoutRateScheduler(callbacks.Callback):
    """Custom callback to adjust dropout rate during training"""
    
    def __init__(self, dropout_start=0.5, dropout_end=0.3, epochs=50):
        super().__init__()
        self.dropout_start = dropout_start
        self.dropout_end = dropout_end
        self.epochs = epochs
    
    def on_epoch_begin(self, epoch, logs=None):
        # Linear decay of dropout rate
        decay_rate = (self.dropout_start - self.dropout_end) / self.epochs
        new_dropout = self.dropout_start - decay_rate * epoch
        new_rate = max(new_dropout, self.dropout_end)
        
        # Set the dropout rate for both dropout layers
        self.model.get_layer('dropout1').rate = new_rate
        self.model.get_layer('dropout2').rate = new_rate
        print(f"Epoch {epoch+1}: dropout rate set to {new_rate:.4f}")