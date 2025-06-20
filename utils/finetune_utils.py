"""
Fine-tuning utilities for model parameter extraction and preparation.

This module provides functions to:
- Extract model parameters from saved model filenames
- Load pretrained models from disk
- Freeze layers for fine-tuning
- Generate paths for fine-tuned models

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