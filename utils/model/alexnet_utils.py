import tensorflow as tf
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

def adapt_alexnet_for_resolution(model, target_size, dropout_rate=None):
    """
    Adapt a pretrained AlexNet model for a different input resolution.
    
    This function rebuilds the AlexNet architecture with the new input size,
    matching the exact layer structure from build_alexnet().
    
    Args:
        model: Pretrained AlexNet model
        target_size: Target image size (assuming square images)
        dropout_rate: Dropout rate for new model (uses original if None)
        
    Returns:
        Adapted AlexNet model for the new resolution
    """
    # Extract dropout rate from original model if not specified
    if dropout_rate is None:
        try:
            # Get dropout rate from the original model's dropout layers
            dropout_layer = model.get_layer('dropout1')
            dropout_rate = dropout_layer.rate
        except:
            dropout_rate = 0.5  # Default fallback
    
    print(f"Building new AlexNet for {target_size}x{target_size} resolution...")
    print(f"Using dropout rate: {dropout_rate}")
    
    # Use the existing build_alexnet function with new input size
    adapted_model = build_alexnet(
        input_shape=(target_size, target_size, 1),
        num_classes=2,  # For binary pneumonia classification with one-hot encoding
        dropout_rate=dropout_rate
    )
    
    print(f"✓ AlexNet successfully adapted from {model.input_shape} to ({target_size}, {target_size}, 1)")
    print(f"Original parameters: {model.count_params():,}")
    print(f"Adapted parameters: {adapted_model.count_params():,}")
    
    return adapted_model


def transfer_alexnet_weights(source_model, target_model):
    """
    Transfer weights between AlexNet models with potentially different input sizes.
    
    This function matches layers by their position in the AlexNet architecture
    rather than by name, since the layer structure is consistent.
    
    Args:
        source_model: Source AlexNet model
        target_model: Target AlexNet model
        
    Returns:
        Target model with transferred weights
    """
    print("\nTransferring AlexNet weights...")
    
    # Define the layer types we expect in AlexNet in order
    expected_layer_types = [
        'InputLayer',      # 0
        'Conv2D',          # 1 - First conv
        'BatchNormalization', # 2
        'Activation',      # 3
        'MaxPooling2D',    # 4
        'Conv2D',          # 5 - Second conv
        'BatchNormalization', # 6
        'Activation',      # 7
        'MaxPooling2D',    # 8
        'Conv2D',          # 9 - Third conv
        'BatchNormalization', # 10
        'Activation',      # 11
        'Conv2D',          # 12 - Fourth conv
        'BatchNormalization', # 13
        'Activation',      # 14
        'Conv2D',          # 15 - Fifth conv
        'BatchNormalization', # 16
        'Activation',      # 17
        'MaxPooling2D',    # 18
        'Flatten',         # 19
        'Dense',           # 20 - First dense
        'BatchNormalization', # 21
        'Activation',      # 22
        'Dropout',         # 23
        'Dense',           # 24 - Second dense
        'BatchNormalization', # 25
        'Activation',      # 26
        'Dropout',         # 27
        'Dense'            # 28 - Output layer
    ]
    
    transferred_count = 0
    
    # Transfer weights for compatible layers (Conv2D, Dense, BatchNorm)
    for i, (src_layer, tgt_layer) in enumerate(zip(source_model.layers, target_model.layers)):
        layer_type = type(src_layer).__name__
        
        # Only transfer weights for layers that have them
        if hasattr(src_layer, 'get_weights') and len(src_layer.get_weights()) > 0:
            
            if isinstance(src_layer, tf.keras.layers.Conv2D):
                # Transfer Conv2D weights (should work for same filter sizes)
                try:
                    if (src_layer.filters == tgt_layer.filters and 
                        src_layer.kernel_size == tgt_layer.kernel_size):
                        tgt_layer.set_weights(src_layer.get_weights())
                        print(f"✓ Conv2D layer {i}: {src_layer.filters} filters")
                        transferred_count += 1
                    else:
                        print(f"✗ Conv2D layer {i}: incompatible ({src_layer.filters} vs {tgt_layer.filters} filters)")
                except Exception as e:
                    print(f"✗ Conv2D layer {i}: transfer failed - {e}")
                        
            elif isinstance(src_layer, tf.keras.layers.Dense):
                # For Dense layers, only transfer if not the final layer or if dimensions match
                try:
                    src_weights = src_layer.get_weights()
                    tgt_weights = tgt_layer.get_weights()
                    
                    # Skip output layer if dimensions don't match (num_classes difference)
                    if i == len(source_model.layers) - 1:  # Output layer
                        if src_weights[0].shape[1] != tgt_weights[0].shape[1]:
                            print(f"✗ Dense layer {i} (output): skipping due to different output dimensions")
                            continue
                    
                    # Check if weight dimensions match
                    if src_weights[0].shape == tgt_weights[0].shape:
                        tgt_layer.set_weights(src_weights)
                        print(f"✓ Dense layer {i}: {src_layer.units} units")
                        transferred_count += 1
                    else:
                        print(f"✗ Dense layer {i}: incompatible shapes {src_weights[0].shape} vs {tgt_weights[0].shape}")
                except Exception as e:
                    print(f"✗ Dense layer {i}: transfer failed - {e}")
                        
            elif isinstance(src_layer, tf.keras.layers.BatchNormalization):
                # Transfer BatchNorm weights
                try:
                    tgt_layer.set_weights(src_layer.get_weights())
                    print(f"✓ BatchNorm layer {i}")
                    transferred_count += 1
                except Exception as e:
                    print(f"✗ BatchNorm layer {i}: transfer failed - {e}")
    
    print(f"\nAlexNet weight transfer complete: {transferred_count} layers transferred")
    
    return target_model