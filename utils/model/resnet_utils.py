import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

#### ResNet-18 Functions ####

def residual_block(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
    """Residual block for ResNet models"""
    # Shortcut connection
    shortcut = x
    
    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=use_bias, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation('relu', name=f"{name}_relu1")(x)
    
    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size, padding='same', use_bias=use_bias, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    
    # If stride > 1 or channels changed, we need to downsample the shortcut connection
    if stride > 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=use_bias, name=f"{name}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)
    
    # Add shortcut to main path
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation('relu', name=f"{name}_relu2")(x)
    return x

def build_resnet18(input_shape=(28, 28, 1), num_classes=2, dropout_rate=0.3):
    """
    Build a ResNet-18 model adapted for small input size (28x28) with dropout
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution - reduced kernel and stride due to small input size
    x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=True, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation('relu', name="relu1")(x)
    x = layers.MaxPooling2D(2, strides=2, padding='same', name="maxpool1")(x)
    
    # ResNet blocks
    # Stage 1
    x = residual_block(x, 64, name="stage1_block1")
    x = residual_block(x, 64, name="stage1_block2")
    
    # Stage 2
    x = residual_block(x, 128, stride=2, name="stage2_block1")
    x = residual_block(x, 128, name="stage2_block2")
    
    # Stage 3
    x = residual_block(x, 256, stride=2, name="stage3_block1")
    x = residual_block(x, 256, name="stage3_block2")
    
    # Stage 4
    x = residual_block(x, 512, stride=2, name="stage4_block1")
    x = residual_block(x, 512, name="stage4_block2")
    
    # Final layers
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)  # Add dropout layer
    outputs = layers.Dense(num_classes, activation='softmax', name="fc")(x)
    
    model = models.Model(inputs, outputs)
    return model

#### ResNet-50 Functions ####

def bottleneck_block(x, filters, kernel_size=3, stride=1, use_bias=True, name=None):
    """Bottleneck block for ResNet-50"""
    shortcut = x
    
    # First 1x1 convolution to reduce dimensions
    x = layers.Conv2D(filters, 1, strides=1, padding='same', use_bias=use_bias, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation('relu', name=f"{name}_relu1")(x)
    
    # 3x3 convolution
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=use_bias, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation('relu', name=f"{name}_relu2")(x)
    
    # 1x1 convolution to expand dimensions
    x = layers.Conv2D(filters * 4, 1, strides=1, padding='same', use_bias=use_bias, name=f"{name}_conv3")(x)
    x = layers.BatchNormalization(name=f"{name}_bn3")(x)
    
    # If stride > 1 or channels changed, we need to downsample the shortcut connection
    if stride > 1 or shortcut.shape[-1] != filters * 4:
        shortcut = layers.Conv2D(filters * 4, 1, strides=stride, padding='same', use_bias=use_bias, name=f"{name}_shortcut_conv")(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_shortcut_bn")(shortcut)
    
    # Add shortcut to main path
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.Activation('relu', name=f"{name}_relu3")(x)
    return x

def build_resnet50(input_shape=(28, 28, 1), num_classes=2, dropout_rate=0.3):
    """
    Build a ResNet-50 model with dropout
    
    Args:
        input_shape: Input shape of the images
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        
    Returns:
        model: ResNet-50 model
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution
    x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=True, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.Activation('relu', name="relu1")(x)
    x = layers.MaxPooling2D(2, strides=2, padding='same', name="maxpool1")(x)
    
    # Stage 1 (3 bottleneck blocks)
    x = bottleneck_block(x, 64, name="stage1_block1")
    x = bottleneck_block(x, 64, name="stage1_block2")
    x = bottleneck_block(x, 64, name="stage1_block3")
    
    # Stage 2 (4 bottleneck blocks)
    x = bottleneck_block(x, 128, stride=2, name="stage2_block1")
    x = bottleneck_block(x, 128, name="stage2_block2")
    x = bottleneck_block(x, 128, name="stage2_block3")
    x = bottleneck_block(x, 128, name="stage2_block4")
    
    # Stage 3 (6 bottleneck blocks)
    x = bottleneck_block(x, 256, stride=2, name="stage3_block1")
    x = bottleneck_block(x, 256, name="stage3_block2")
    x = bottleneck_block(x, 256, name="stage3_block3")
    x = bottleneck_block(x, 256, name="stage3_block4")
    x = bottleneck_block(x, 256, name="stage3_block5")
    x = bottleneck_block(x, 256, name="stage3_block6")
    
    # Stage 4 (3 bottleneck blocks)
    x = bottleneck_block(x, 512, stride=2, name="stage4_block1")
    x = bottleneck_block(x, 512, name="stage4_block2")
    x = bottleneck_block(x, 512, name="stage4_block3")
    
    # Final layers - same structure as ResNet-18 for compatibility
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)  # Same dropout layer name for compatibility
    outputs = layers.Dense(num_classes, activation='softmax', name="fc")(x)
    
    model = models.Model(inputs, outputs)
    return model

#### ResNet18 Adaptation and Weight Transfer Functions ####

def adapt_resnet18_for_resolution(model, target_size, dropout_rate=None):
    """
    Adapt a pretrained ResNet18 model for a different input resolution.
    
    This function rebuilds the ResNet18 architecture with the new input size,
    matching the exact layer structure from build_resnet18().
    
    Args:
        model: Pretrained ResNet18 model
        target_size: Target image size (assuming square images)
        dropout_rate: Dropout rate for new model (uses original if None)
        
    Returns:
        Adapted ResNet18 model for the new resolution
    """
    # Extract dropout rate from original model if not specified
    if dropout_rate is None:
        try:
            # Get dropout rate from the original model's dropout layer
            dropout_layer = model.get_layer('dropout')
            dropout_rate = dropout_layer.rate
        except:
            dropout_rate = 0.3  # Default fallback
    
    print(f"Building new ResNet18 for {target_size}x{target_size} resolution...")
    print(f"Using dropout rate: {dropout_rate}")
    
    # Use the existing build_resnet18 function with new input size
    adapted_model = build_resnet18(
        input_shape=(target_size, target_size, 1),
        num_classes=2,  # For binary pneumonia classification with one-hot encoding
        dropout_rate=dropout_rate
    )
    
    print(f"✓ ResNet18 successfully adapted from {model.input_shape} to ({target_size}, {target_size}, 1)")
    print(f"Original parameters: {model.count_params():,}")
    print(f"Adapted parameters: {adapted_model.count_params():,}")
    
    return adapted_model


def transfer_resnet18_weights(source_model, target_model):
    """
    Transfer weights between ResNet18 models with potentially different input sizes.
    
    This function matches layers by their name and type, since ResNet18 has a 
    consistent naming convention for layers.
    
    Args:
        source_model: Source ResNet18 model
        target_model: Target ResNet18 model
        
    Returns:
        Target model with transferred weights
    """
    print("\nTransferring ResNet18 weights...")
    
    transferred_count = 0
    
    # Create a mapping of source layers by name
    source_layers = {layer.name: layer for layer in source_model.layers}
    
    # Transfer weights for compatible layers
    for target_layer in target_model.layers:
        layer_name = target_layer.name
        
        # Skip if source doesn't have this layer
        if layer_name not in source_layers:
            continue
            
        source_layer = source_layers[layer_name]
        
        # Only transfer weights for layers that have them
        if hasattr(source_layer, 'get_weights') and len(source_layer.get_weights()) > 0:
            
            if isinstance(source_layer, tf.keras.layers.Conv2D):
                # Transfer Conv2D weights (should work for same filter configurations)
                try:
                    if (source_layer.filters == target_layer.filters and 
                        source_layer.kernel_size == target_layer.kernel_size):
                        target_layer.set_weights(source_layer.get_weights())
                        print(f"✓ Conv2D layer {layer_name}: {source_layer.filters} filters")
                        transferred_count += 1
                    else:
                        print(f"✗ Conv2D layer {layer_name}: incompatible configuration")
                except Exception as e:
                    print(f"✗ Conv2D layer {layer_name}: transfer failed - {e}")
                        
            elif isinstance(source_layer, tf.keras.layers.Dense):
                # For Dense layers, only transfer if dimensions match
                try:
                    source_weights = source_layer.get_weights()
                    target_weights = target_layer.get_weights()
                    
                    # Skip output layer if dimensions don't match (num_classes difference)
                    if layer_name == "fc":  # Output layer
                        if source_weights[0].shape[1] != target_weights[0].shape[1]:
                            print(f"✗ Dense layer {layer_name} (output): skipping due to different output dimensions")
                            continue
                    
                    # Check if weight dimensions match
                    if source_weights[0].shape == target_weights[0].shape:
                        target_layer.set_weights(source_weights)
                        print(f"✓ Dense layer {layer_name}: {source_layer.units} units")
                        transferred_count += 1
                    else:
                        print(f"✗ Dense layer {layer_name}: incompatible shapes {source_weights[0].shape} vs {target_weights[0].shape}")
                except Exception as e:
                    print(f"✗ Dense layer {layer_name}: transfer failed - {e}")
                        
            elif isinstance(source_layer, tf.keras.layers.BatchNormalization):
                # Transfer BatchNorm weights
                try:
                    target_layer.set_weights(source_layer.get_weights())
                    print(f"✓ BatchNorm layer {layer_name}")
                    transferred_count += 1
                except Exception as e:
                    print(f"✗ BatchNorm layer {layer_name}: transfer failed - {e}")
    
    print(f"\nResNet18 weight transfer complete: {transferred_count} layers transferred")
    
    return target_model

#### ResNet50 Adaptation and Weight Transfer Functions ####

def adapt_resnet50_for_resolution(model, target_size, dropout_rate=None):
    """
    Adapt a pretrained ResNet50 model for a different input resolution.
    
    This function rebuilds the ResNet50 architecture with the new input size,
    matching the exact layer structure from build_resnet50().
    
    Args:
        model: Pretrained ResNet50 model
        target_size: Target image size (assuming square images)
        dropout_rate: Dropout rate for new model (uses original if None)
        
    Returns:
        Adapted ResNet50 model for the new resolution
    """
    # Extract dropout rate from original model if not specified
    if dropout_rate is None:
        try:
            # Get dropout rate from the original model's dropout layer
            dropout_layer = model.get_layer('dropout')
            dropout_rate = dropout_layer.rate
        except:
            dropout_rate = 0.3  # Default fallback
    
    print(f"Building new ResNet50 for {target_size}x{target_size} resolution...")
    print(f"Using dropout rate: {dropout_rate}")
    
    # Use the existing build_resnet50 function with new input size
    adapted_model = build_resnet50(
        input_shape=(target_size, target_size, 1),
        num_classes=2,  # For binary pneumonia classification with one-hot encoding
        dropout_rate=dropout_rate
    )
    
    print(f"✓ ResNet50 successfully adapted from {model.input_shape} to ({target_size}, {target_size}, 1)")
    print(f"Original parameters: {model.count_params():,}")
    print(f"Adapted parameters: {adapted_model.count_params():,}")
    
    return adapted_model


def transfer_resnet50_weights(source_model, target_model):
    """
    Transfer weights between ResNet50 models with potentially different input sizes.
    
    This function matches layers by their name and type, since ResNet50 has a 
    consistent naming convention for layers.
    
    Args:
        source_model: Source ResNet50 model
        target_model: Target ResNet50 model
        
    Returns:
        Target model with transferred weights
    """
    print("\nTransferring ResNet50 weights...")
    
    transferred_count = 0
    
    # Create a mapping of source layers by name
    source_layers = {layer.name: layer for layer in source_model.layers}
    
    # Transfer weights for compatible layers
    for target_layer in target_model.layers:
        layer_name = target_layer.name
        
        # Skip if source doesn't have this layer
        if layer_name not in source_layers:
            continue
            
        source_layer = source_layers[layer_name]
        
        # Only transfer weights for layers that have them
        if hasattr(source_layer, 'get_weights') and len(source_layer.get_weights()) > 0:
            
            if isinstance(source_layer, tf.keras.layers.Conv2D):
                # Transfer Conv2D weights (should work for same filter configurations)
                try:
                    if (source_layer.filters == target_layer.filters and 
                        source_layer.kernel_size == target_layer.kernel_size):
                        target_layer.set_weights(source_layer.get_weights())
                        print(f"✓ Conv2D layer {layer_name}: {source_layer.filters} filters")
                        transferred_count += 1
                    else:
                        print(f"✗ Conv2D layer {layer_name}: incompatible configuration")
                except Exception as e:
                    print(f"✗ Conv2D layer {layer_name}: transfer failed - {e}")
                        
            elif isinstance(source_layer, tf.keras.layers.Dense):
                # For Dense layers, only transfer if dimensions match
                try:
                    source_weights = source_layer.get_weights()
                    target_weights = target_layer.get_weights()
                    
                    # Skip output layer if dimensions don't match (num_classes difference)
                    if layer_name == "fc":  # Output layer
                        if source_weights[0].shape[1] != target_weights[0].shape[1]:
                            print(f"✗ Dense layer {layer_name} (output): skipping due to different output dimensions")
                            continue
                    
                    # Check if weight dimensions match
                    if source_weights[0].shape == target_weights[0].shape:
                        target_layer.set_weights(source_weights)
                        print(f"✓ Dense layer {layer_name}: {source_layer.units} units")
                        transferred_count += 1
                    else:
                        print(f"✗ Dense layer {layer_name}: incompatible shapes {source_weights[0].shape} vs {target_weights[0].shape}")
                except Exception as e:
                    print(f"✗ Dense layer {layer_name}: transfer failed - {e}")
                        
            elif isinstance(source_layer, tf.keras.layers.BatchNormalization):
                # Transfer BatchNorm weights
                try:
                    target_layer.set_weights(source_layer.get_weights())
                    print(f"✓ BatchNorm layer {layer_name}")
                    transferred_count += 1
                except Exception as e:
                    print(f"✗ BatchNorm layer {layer_name}: transfer failed - {e}")
    
    print(f"\nResNet50 weight transfer complete: {transferred_count} layers transferred")
    
    return target_model

#### Custom Callbacks ####

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
        
        # Set the dropout rate for the dropout layer
        self.model.get_layer('dropout').rate = max(new_dropout, self.dropout_end)
        print(f"Epoch {epoch+1}: dropout rate set to {self.model.get_layer('dropout').rate:.4f}")
