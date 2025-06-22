"""
Core XAI Methods for Pneumonia Detection Models
Provides GradCAM, LIME, and SHAP analysis functions
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Import XAI libraries
import lime
from lime import lime_image
import shap
from tf_explain.core.grad_cam import GradCAM

# Try to import skimage for resizing, if not available use tf
try:
    from skimage.transform import resize
except ImportError:
    def resize(image, output_shape):
        return tf.image.resize(image, output_shape).numpy()


def apply_gradcam_analysis(model, images, class_indices, layer_name=None):
    """Apply GradCAM to understand which parts of images the model focuses on."""
    
    # Find the last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                layer_name = layer.name
                break
    
    if layer_name is None:
        print("No convolutional layer found for GradCAM")
        return [None] * len(images), None
    
    print(f"Using layer '{layer_name}' for GradCAM analysis")
    
    # Initialize GradCAM
    gradcam = GradCAM()
    
    # Generate GradCAM for each image
    gradcam_results = []
    
    for i, (img, class_idx) in enumerate(zip(images, class_indices)):
        try:
            # Generate GradCAM heatmap
            heatmap = gradcam.explain(
                validation_data=(np.expand_dims(img, 0), None),
                model=model,
                layer_name=layer_name,
                class_index=class_idx
            )
            gradcam_results.append(heatmap)
        except Exception as e:
            print(f"GradCAM failed for image {i}: {e}")
            gradcam_results.append(None)
    
    return gradcam_results, layer_name


def apply_lime_analysis(model, images, num_samples=1000):
    """Apply LIME to explain model predictions."""
    
    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Create prediction function for LIME
    def predict_fn(images):
        # Ensure images have the right shape
        if len(images.shape) == 3:
            images = np.expand_dims(images, -1)
        return model.predict(images, verbose=0)
    
    lime_results = []
    
    for i, img in enumerate(images):
        try:
            print(f"Processing LIME for image {i+1}/{len(images)}...")
            
            # Generate LIME explanation
            explanation = explainer.explain_instance(
                img.squeeze(),
                predict_fn,
                top_labels=2,
                hide_color=0,
                num_samples=num_samples
            )
            
            lime_results.append(explanation)
            
        except Exception as e:
            print(f"LIME failed for image {i}: {e}")
            lime_results.append(None)
    
    return lime_results


def apply_shap_analysis(model, images, background_samples=50):
    """Apply SHAP to explain model predictions."""
    
    try:
        # Create background dataset for SHAP
        background = images[:background_samples]
        
        # Initialize SHAP explainer
        explainer = shap.DeepExplainer(model, background)
        
        # Generate SHAP values
        print(f"Computing SHAP values for {len(images)} images...")
        shap_values = explainer.shap_values(images)
        
        return shap_values
        
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        return None
