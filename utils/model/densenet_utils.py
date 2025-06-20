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

#### DenseNet169 Adaptation and Weight Transfer Functions ####

def adapt_densenet169_for_resolution(model, target_size, dropout_rate=None):
    """
    Adapt a pretrained DenseNet169 model for a different input resolution.
    
    This function rebuilds the DenseNet169 architecture with the new input size,
    keeping the same structure but adapting the preprocessing layers.
    
    Args:
        model: Pretrained DenseNet169 model
        target_size: Target image size (assuming square images)
        dropout_rate: Dropout rate for new model (uses original if None)
        
    Returns:
        Adapted DenseNet169 model for the new resolution
    """
    # Extract dropout rate from original model if not specified
    if dropout_rate is None:
        try:
            # Find the dropout layer in the original model
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    dropout_rate = layer.rate
                    break
            if dropout_rate is None:
                dropout_rate = 0.5  # Default fallback
        except:
            dropout_rate = 0.5  # Default fallback
    
    print(f"Building new DenseNet169 for {target_size}x{target_size} resolution...")
    print(f"Using dropout rate: {dropout_rate}")
    
    # Use the existing build_densenet169_model function with new input size
    adapted_model = build_densenet169_model(
        input_shape=(target_size, target_size, 1),
        num_classes=2,  # For binary pneumonia classification
        dropout_rate=dropout_rate
    )
    
    print(f"✓ DenseNet169 successfully adapted from {model.input_shape} to ({target_size}, {target_size}, 1)")
    print(f"Original parameters: {model.count_params():,}")
    print(f"Adapted parameters: {adapted_model.count_params():,}")
    
    return adapted_model


def transfer_densenet169_weights(source_model, target_model):
    """
    Transfer weights between DenseNet169 models with potentially different input sizes.
    
    Since DenseNet169 uses TensorFlow's pre-trained model, weight transfer is more complex.
    This function will transfer weights from the custom classification head layers.
    
    Args:
        source_model: Source DenseNet169 model
        target_model: Target DenseNet169 model
        
    Returns:
        Target model with transferred weights
    """
    print("\nTransferring DenseNet169 weights...")
    
    transferred_count = 0
    
    # Create a mapping of source layers by name for custom layers
    source_layers = {layer.name: layer for layer in source_model.layers}
    
    # Transfer weights for compatible custom layers (not the pre-trained DenseNet base)
    for target_layer in target_model.layers:
        layer_name = target_layer.name
        
        # Skip if source doesn't have this layer
        if layer_name not in source_layers:
            continue
            
        source_layer = source_layers[layer_name]
        
        # Only transfer weights for layers that have them and are not pre-trained
        if (hasattr(source_layer, 'get_weights') and 
            len(source_layer.get_weights()) > 0 and
            not layer_name.startswith('densenet169')):  # Skip pre-trained base model layers
            
            try:
                # Check if the layer configurations match
                if (hasattr(source_layer, 'units') and hasattr(target_layer, 'units') and
                    isinstance(source_layer, tf.keras.layers.Dense)):
                    # Transfer Dense layer weights
                    if source_layer.units == target_layer.units:
                        target_layer.set_weights(source_layer.get_weights())
                        print(f"✓ Dense layer {layer_name}: {source_layer.units} units")
                        transferred_count += 1
                    else:
                        print(f"✗ Dense layer {layer_name}: incompatible units {source_layer.units} vs {target_layer.units}")
                        
                elif isinstance(source_layer, tf.keras.layers.BatchNormalization):
                    # Transfer BatchNorm weights
                    target_layer.set_weights(source_layer.get_weights())
                    print(f"✓ BatchNorm layer {layer_name}")
                    transferred_count += 1
                    
                elif isinstance(source_layer, tf.keras.layers.Conv2D):
                    # Transfer custom Conv2D weights (like the grayscale to RGB conversion layer)
                    source_weights = source_layer.get_weights()
                    target_weights = target_layer.get_weights()
                    
                    if len(source_weights) == len(target_weights):
                        # Check if shapes match
                        shapes_match = all(sw.shape == tw.shape for sw, tw in zip(source_weights, target_weights))
                        if shapes_match:
                            target_layer.set_weights(source_weights)
                            print(f"✓ Conv2D layer {layer_name}")
                            transferred_count += 1
                        else:
                            print(f"✗ Conv2D layer {layer_name}: incompatible shapes")
                    else:
                        print(f"✗ Conv2D layer {layer_name}: different number of weight arrays")
                        
                elif isinstance(source_layer, tf.keras.layers.Dropout):
                    # For Dropout layers, we don't transfer weights but ensure the rate matches
                    target_layer.rate = source_layer.rate
                    print(f"✓ Dropout layer {layer_name}: rate set to {source_layer.rate}")
                    transferred_count += 1
                    
            except Exception as e:
                print(f"✗ Layer {layer_name}: transfer failed - {e}")
    
    print(f"\nDenseNet169 weight transfer complete: {transferred_count} layers transferred")
    print("Note: Pre-trained DenseNet169 base model weights are loaded from ImageNet automatically")
    
    return target_model