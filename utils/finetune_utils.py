"""
Fine-tuning utilities for adapting pre-trained models to different image resolutions.

This module provides functions to:
- Extract model parameters from saved model filenames
- Load pretrained models and adapt them for different input resolutions
- Handle model architecture modifications for fine-tuning

Authors: Rafaela Abrunhosa, Miguel Pinto
"""

import os
import re
import tensorflow as tf
from typing import Dict, Tuple, Any


def extract_model_params(model_path: str) -> Dict[str, Any]:
    """
    Extract training parameters from a saved model filename.
    
    Expected filename format: 
    modelname_b{batch_size}_lr{learning_rate}_dr{dropout_rate}_pneumonia_model.keras
    or
    modelname_b{batch_size}_lr{learning_rate}_dr{dropout_start}_{dropout_end}_pneumonia_model.keras
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        Dict[str, Any]: Dictionary containing extracted parameters
    """
    filename = os.path.basename(model_path)
    
    # Initialize default parameters
    params = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'dropout_rate': 0.3,
        'augmented': False
    }
    
    try:
        # Check if model is augmented
        if 'augmented' in filename:
            params['augmented'] = True
        
        # Extract batch size
        batch_match = re.search(r'_b(\d+)_', filename)
        if batch_match:
            params['batch_size'] = int(batch_match.group(1))
        
        # Extract learning rate
        lr_match = re.search(r'_lr([\d.]+)_', filename)
        if lr_match:
            params['learning_rate'] = float(lr_match.group(1))
        
        # Extract dropout rate(s)
        # First try to match custom scheduler pattern (two values)
        dropout_custom_match = re.search(r'_dr([\d.]+)_([\d.]+)_', filename)
        if dropout_custom_match:
            # For custom scheduler, use the end value as the main dropout rate
            params['dropout_rate'] = float(dropout_custom_match.group(2))
            params['dropout_start'] = float(dropout_custom_match.group(1))
            params['dropout_end'] = float(dropout_custom_match.group(2))
            params['dropout_mode'] = 'custom'
        else:
            # Try to match fixed dropout pattern (single value)
            dropout_match = re.search(r'_dr([\d.]+)_', filename)
            if dropout_match:
                params['dropout_rate'] = float(dropout_match.group(1))
                params['dropout_mode'] = 'fixed'
    
    except Exception as e:
        print(f"Warning: Could not extract all parameters from filename. Using defaults. Error: {e}")
    
    return params


def load_pretrained_model(model_path: str) -> tf.keras.Model:
    """
    Load a pretrained model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        tf.keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded successfully: {model.name}")
    return model


def adapt_model_for_resolution(model: tf.keras.Model, target_size: int, architecture: str = None, dropout_rate: float = None) -> tf.keras.Model:
    """
    Adapt a pretrained model for a different input resolution.
    
    This is a general-purpose function that delegates to architecture-specific
    adaptation functions for best results.
    
    Args:
        model (tf.keras.Model): Pretrained model
        target_size (int): Target image size (assuming square images)
        architecture (str): Architecture name ('alexnet', 'resnet18', etc.). If None, tries to infer from model name
        dropout_rate (float, optional): Dropout rate for new model
        
    Returns:
        tf.keras.Model: Adapted model for the new resolution
    """
    # Use provided architecture or try to determine from model name
    if architecture is None:
        model_name = model.name.lower()
        if 'alexnet' in model_name:
            architecture = 'alexnet'
        elif 'resnet18' in model_name or 'resnet_18' in model_name:
            architecture = 'resnet_18'
        elif 'resnet50' in model_name or 'resnet_50' in model_name:
            architecture = 'resnet_50'
        elif 'densenet' in model_name:
            architecture = 'densenet_169'
    
    if architecture == 'alexnet':
        # Use AlexNet-specific adapter
        try:
            from utils.model.alexnet_utils import adapt_alexnet_for_resolution
            return adapt_alexnet_for_resolution(model, target_size, dropout_rate)
        except ImportError:
            print("Warning: AlexNet utils not found, using generic adaptation")
    
    elif architecture == 'resnet_18':
        # Use ResNet18-specific adapter
        try:
            from utils.model.resnet_utils import adapt_resnet18_for_resolution
            return adapt_resnet18_for_resolution(model, target_size, dropout_rate)
        except ImportError:
            print("Warning: ResNet18 adapter not implemented yet")
            return None

    elif architecture == 'resnet_50':
        # Use ResNet50-specific adapter
        try:
            from utils.model.resnet_utils import adapt_resnet50_for_resolution
            return adapt_resnet50_for_resolution(model, target_size, dropout_rate)
        except ImportError:
            print("Warning: ResNet50 adapter not implemented yet")
            return None

    elif architecture == 'densenet_169':
        # Use DenseNet-specific adapter (when implemented)
        try:
            from utils.model.densenet_utils import adapt_densenet_for_resolution
            return adapt_densenet_for_resolution(model, target_size, dropout_rate)
        except ImportError:
            print("Warning: DenseNet adapter not implemented yet")
    
    # Fallback: generic layer-by-layer adaptation
    print("Using generic layer-by-layer adaptation...")
    return _generic_adapt_model(model, target_size, dropout_rate)


def _generic_adapt_model(model: tf.keras.Model, target_size: int, dropout_rate: float = None) -> tf.keras.Model:
    """
    Generic layer-by-layer model adaptation fallback.
    Only use when architecture-specific adapters are not available.
    """
    # Get the model architecture details
    original_input_shape = model.input_shape
    print(f"Original input shape: {original_input_shape}")
    print(f"Target input shape: ({target_size}, {target_size}, 1)")
    
    # Create new input layer with target size
    new_input = tf.keras.layers.Input(shape=(target_size, target_size, 1))
    x = new_input
    
    # Rebuild the architecture by examining layer types and configurations
    # This is a simplified approach that works for basic CNN architectures
    
    conv_layer_count = 0
    dense_layer_count = 0
    
    for i, layer in enumerate(model.layers[1:], 1):  # Skip the input layer
        layer_type = type(layer).__name__
        
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_layer_count += 1
            # Create new Conv2D layer with same configuration
            x = tf.keras.layers.Conv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation,
                name=f"conv2d_{conv_layer_count}_adapted"
            )(x)
            
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            # Create new MaxPooling2D layer
            x = tf.keras.layers.MaxPooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                name=f"maxpool_{conv_layer_count}_adapted"
            )(x)
            
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            # Create new BatchNormalization layer
            x = tf.keras.layers.BatchNormalization(
                name=f"batchnorm_{conv_layer_count}_adapted"
            )(x)
            
        elif isinstance(layer, tf.keras.layers.Activation):
            # Create new Activation layer
            x = tf.keras.layers.Activation(
                activation=layer.activation,
                name=f"activation_{conv_layer_count}_adapted"
            )(x)
            
        elif isinstance(layer, (tf.keras.layers.ReLU, tf.keras.layers.LeakyReLU)):
            # Create new ReLU-type layer
            x = type(layer)(name=f"relu_{conv_layer_count}_adapted")(x)
            
        elif isinstance(layer, tf.keras.layers.Flatten):
            # Flatten layer
            x = tf.keras.layers.Flatten(name="flatten_adapted")(x)
            
        elif isinstance(layer, tf.keras.layers.Dense):
            dense_layer_count += 1
            # Create new Dense layer
            x = tf.keras.layers.Dense(
                units=layer.units,
                activation=layer.activation,
                name=f"dense_{dense_layer_count}_adapted"
            )(x)
            
        elif isinstance(layer, tf.keras.layers.Dropout):
            # Create new Dropout layer with specified or original rate
            rate = dropout_rate if dropout_rate is not None else layer.rate
            x = tf.keras.layers.Dropout(
                rate=rate,
                name=f"dropout_{dense_layer_count}_adapted"
            )(x)
            
        else:
            print(f"Warning: Skipping unsupported layer type: {layer_type}")
    
    # Create the new model
    adapted_model = tf.keras.Model(inputs=new_input, outputs=x, name=f"{model.name}_adapted")
    
    print(f"Model successfully adapted for {target_size}x{target_size} input")
    print(f"Original model parameters: {model.count_params():,}")
    print(f"Adapted model parameters: {adapted_model.count_params():,}")
    
    return adapted_model


def freeze_layers(model: tf.keras.Model, freeze_until_layer: str = None, freeze_ratio: float = None, architecture: str = None) -> tf.keras.Model:
    """
    Freeze layers in a model for fine-tuning.
    
    Args:
        model (tf.keras.Model): Model to freeze layers in
        freeze_until_layer (str, optional): Name of the layer up to which to freeze
        freeze_ratio (float, optional): Ratio of layers to freeze (0.0 to 1.0)
        architecture (str, optional): Architecture name (for logging purposes)
        
    Returns:
        tf.keras.Model: Model with frozen layers
    """
    if freeze_until_layer:
        # Freeze layers until specified layer
        freeze_flag = True
        for layer in model.layers:
            if freeze_flag:
                layer.trainable = False
                if layer.name == freeze_until_layer:
                    freeze_flag = False
            else:
                layer.trainable = True
    
    elif freeze_ratio:
        # Freeze a ratio of layers
        total_layers = len(model.layers)
        freeze_count = int(total_layers * freeze_ratio)
        
        for i, layer in enumerate(model.layers):
            if i < freeze_count:
                layer.trainable = False
            else:
                layer.trainable = True
    
    # Print summary of frozen/trainable layers
    frozen_count = sum(1 for layer in model.layers if not layer.trainable)
    trainable_count = sum(1 for layer in model.layers if layer.trainable)
    
    print(f"Frozen layers: {frozen_count}")
    print(f"Trainable layers: {trainable_count}")
    
    return model


def generate_tuned_model_path(original_params: Dict[str, Any], 
                             new_image_size: int,
                             output_path: str,
                             architecture: str) -> str:
    """
    Generate the path for the fine-tuned model based on original parameters and new image size.
    
    Args:
        original_params: Dictionary containing original model parameters
        new_image_size: New image size for fine-tuning
        output_path: Base output path for models
        architecture: Architecture name (e.g., 'alexnet')
        
    Returns:
        Path for the fine-tuned model
    """
    augmented_str = "_augmented" if original_params['augmented'] else ""
    batch_size = original_params['batch_size']
    learning_rate = original_params['learning_rate']
    dropout_rate = original_params['dropout_rate']
    
    
    filename = f"{architecture}_tuned{augmented_str}_b{batch_size}_lr{learning_rate}_dr{dropout_rate}_{new_image_size}px_pneumonia_model.keras"
    
    return os.path.join(output_path, filename)


def transfer_compatible_weights(source_model: tf.keras.Model, target_model: tf.keras.Model, architecture: str = None) -> tf.keras.Model:
    """
    Transfer weights from source model to target model for compatible layers.
    
    This function delegates to architecture-specific transfer functions when available,
    or falls back to a generic approach.
    
    Args:
        source_model: Model to transfer weights from
        target_model: Model to transfer weights to
        architecture: Architecture name ('alexnet', 'resnet18', etc.). If None, tries to infer from model names
        
    Returns:
        Target model with transferred weights
    """
    # Use provided architecture or try to determine from model names
    if architecture is None:
        source_name = source_model.name.lower()
        target_name = target_model.name.lower()
        
        if 'alexnet' in source_name and 'alexnet' in target_name:
            architecture = 'alexnet'
        elif ('resnet18' in source_name or 'resnet_18' in source_name) and ('resnet18' in target_name or 'resnet_18' in target_name):
            architecture = 'resnet_18'
        elif ('resnet50' in source_name or 'resnet_50' in source_name) and ('resnet50' in target_name or 'resnet_50' in target_name):
            architecture = 'resnet_50'
        elif 'densenet' in source_name and 'densenet' in target_name:
            architecture = 'densenet_169'
    
    # Use architecture-specific transfer function
    if architecture == 'alexnet':
        try:
            from utils.model.alexnet_utils import transfer_alexnet_weights
            return transfer_alexnet_weights(source_model, target_model)
        except ImportError:
            print("Warning: AlexNet transfer function not found, using generic transfer")
    
    elif architecture == 'resnet_18':
        try:
            from utils.model.resnet_utils import transfer_resnet18_weights
            return transfer_resnet18_weights(source_model, target_model)
        except ImportError:
            print("Warning: ResNet18 transfer function not implemented yet")
            return target_model

    elif architecture == 'resnet_50':
        try:
            from utils.model.resnet_utils import transfer_resnet50_weights
            return transfer_resnet50_weights(source_model, target_model)
        except ImportError:
            print("Warning: ResNet50 transfer function not implemented yet")
            return target_model

    elif architecture == 'densenet_169':
        try:
            from utils.model.densenet_utils import transfer_densenet_weights
            return transfer_densenet_weights(source_model, target_model)
        except ImportError:
            print("Warning: DenseNet transfer function not implemented yet")
    
    # Fallback to generic transfer
    print("Using generic weight transfer...")
    return _generic_transfer_weights(source_model, target_model)


def _generic_transfer_weights(source_model: tf.keras.Model, target_model: tf.keras.Model) -> tf.keras.Model:
    """
    Generic weight transfer fallback.
    Only transfers weights for layers with identical configurations.
    """
    print("\nGeneric weight transfer...")
    
    source_layers = [layer for layer in source_model.layers if hasattr(layer, 'get_weights') and len(layer.get_weights()) > 0]
    target_layers = [layer for layer in target_model.layers if hasattr(layer, 'set_weights')]
    
    transferred_count = 0
    
    # Try to match layers by type and configuration
    for src_layer in source_layers:
        for tgt_layer in target_layers:
            if (type(src_layer) == type(tgt_layer) and 
                hasattr(src_layer, 'filters') and hasattr(tgt_layer, 'filters') and
                src_layer.filters == tgt_layer.filters and
                src_layer.kernel_size == tgt_layer.kernel_size):
                
                try:
                    tgt_layer.set_weights(src_layer.get_weights())
                    print(f"✓ Transferred weights: {type(src_layer).__name__} ({src_layer.filters} filters)")
                    transferred_count += 1
                    break
                except Exception as e:
                    print(f"✗ Failed to transfer {type(src_layer).__name__}: {e}")
            
            elif (type(src_layer) == type(tgt_layer) and 
                  hasattr(src_layer, 'units') and hasattr(tgt_layer, 'units') and
                  src_layer.units == tgt_layer.units):
                
                try:
                    # For Dense layers, only transfer if dimensions match
                    src_weights = src_layer.get_weights()
                    if len(src_weights) > 0:
                        # Check if weight shapes are compatible
                        tgt_weights_shape = tgt_layer.get_weights()
                        if len(tgt_weights_shape) > 0 and src_weights[0].shape == tgt_weights_shape[0].shape:
                            tgt_layer.set_weights(src_weights)
                            print(f"✓ Transferred weights: {type(src_layer).__name__} ({src_layer.units} units)")
                            transferred_count += 1
                            break
                except Exception as e:
                    print(f"✗ Failed to transfer {type(src_layer).__name__}: {e}")
    
    print(f"\nGeneric weight transfer complete: {transferred_count} layers transferred")
    
    return target_model


def create_architecture_specific_model(architecture: str, input_shape: tuple, num_classes: int = 2, dropout_rate: float = 0.3) -> tf.keras.Model:
    """
    Create a model with specific architecture for the given input shape.
    
    This function imports and uses the existing model builders for different architectures.
    
    Args:
        architecture: Architecture name ('alexnet', 'resnet18', 'resnet50', 'densenet169')
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate to use
        
    Returns:
        Model with the specified architecture
    """
    architecture = architecture.lower()
    
    if architecture == 'alexnet':
        from utils.model.alexnet_utils import build_alexnet
        return build_alexnet(input_shape=input_shape, dropout_rate=dropout_rate)
    
    elif architecture == 'resnet_18':
        from utils.model.resnet_utils import build_resnet18
        return build_resnet18(input_shape=input_shape, dropout_rate=dropout_rate)
    
    elif architecture == 'resnet_50':
        from utils.model.resnet_utils import build_resnet50
        return build_resnet50(input_shape=input_shape, dropout_rate=dropout_rate)
    
    elif architecture == 'densenet_169':
        from utils.model.densenet_utils import build_densenet169_model
        return build_densenet169_model(input_shape=input_shape, dropout_rate=dropout_rate)

    else:
        raise NotImplementedError(f"Architecture {architecture} not implemented. Available: alexnet, resnet_18, resnet_50, densenet_169")
