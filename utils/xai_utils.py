"""
XAI (Explainable AI) Utilities for Pneumonia Detection Models
Main module that imports all XAI functionality from specialized modules
"""

# Import XAI methods
from .xai.xai_methods_utils import (
    apply_gradcam_analysis,
    apply_lime_analysis, 
    apply_shap_analysis
)

# Import data utilities
from .xai.xai_data_utils import (
    parse_model_info,
    load_test_data,
    select_sample_images,
    save_model_predictions,
    load_model_predictions,
    calculate_xai_metrics,
    calculate_xai_similarity_metrics,
    save_xai_analysis_report
)

# Import plotting utilities
from .xai.xai_plots_utils import (
    plot_individual_xai_results,
    plot_xai_comparison_by_architecture,
    plot_xai_comparison_by_resolution,
    analyze_misclassifications
)

# Make all functions available at package level
__all__ = [
    'apply_gradcam_analysis',
    'apply_lime_analysis', 
    'apply_shap_analysis',
    'parse_model_info',
    'load_test_data',
    'select_sample_images',
    'save_model_predictions',
    'load_model_predictions',
    'calculate_xai_metrics',
    'calculate_xai_similarity_metrics',
    'save_xai_analysis_report',
    'plot_individual_xai_results',
    'plot_xai_comparison_by_architecture',
    'plot_xai_comparison_by_resolution',
    'analyze_misclassifications'
]
