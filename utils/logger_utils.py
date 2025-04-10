import json
import os
import numpy as np
from datetime import datetime

class ModelLogger:
    """
    Utility class for logging model training and evaluation information to JSON files.
    Each model run is saved as a separate JSON entry for better organization and retrieval.
    """
    
    def __init__(self):
        """
        Initialize the logger.
        """
        self.current_run = {}
    
    def initialize_run(self, model_info):
        """
        Initialize a new model run with basic information.
        
        Args:
            model_info (dict): Dictionary containing basic model information
        """
        self.current_run = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **model_info
        }
    
    def add_training_info(self, history, execution_time):
        """
        Add training history and execution time information.
        
        Args:
            history: Keras history object
            execution_time (float): Training execution time in seconds
        """
        # Convert history to dictionary if it's not already
        history_dict = history.history if hasattr(history, 'history') else history
        
        # Add history and execution time to current run
        self.current_run.update({
            "training_history": {
                key: [float(val) if isinstance(val, (int, float, np.number)) else val for val in values]
                for key, values in history_dict.items()
            },
            "exec_time": execution_time
        })
    
    def add_evaluation_results(self, results):
        """
        Add evaluation results.
        
        Args:
            results (dict): Dictionary containing evaluation results
        """
        # Make a deep copy of results to avoid modifying the original
        results_copy = {}
        
        # Process each item in results to make it JSON serializable
        for key, value in results.items():
            # Skip large arrays or objects we don't need to save
            if key in ['predictions', 'predicted_classes', 'true_classes']:
                continue
                
            # Handle ROC data specially
            elif key == 'roc':
                results_copy[key] = {
                    'auc': float(results[key]['auc']),
                    'fpr': self._convert_to_serializable(results[key]['fpr']),
                    'tpr': self._convert_to_serializable(results[key]['tpr'])
                }
            
            # Handle confusion matrix
            elif key == 'confusion_matrix':
                results_copy[key] = self._convert_to_serializable(value)
            
            # Handle numeric types
            elif isinstance(value, (int, float, np.number)):
                results_copy[key] = float(value)
                
            # Handle string types
            elif isinstance(value, str):
                results_copy[key] = value
                
            # Handle other types as needed
            else:
                try:
                    # Try to convert to a serializable format
                    results_copy[key] = self._convert_to_serializable(value)
                except:
                    # If conversion fails, skip this item
                    print(f"Warning: Could not serialize '{key}' of type {type(value).__name__}. Skipping.")
        
        # Add evaluation results to current run
        self.current_run["evaluation"] = results_copy
    
    def _convert_to_serializable(self, obj):
        """
        Convert objects to JSON serializable format.
        
        Args:
            obj: The object to convert
            
        Returns:
            A JSON serializable version of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                             np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            # Try to convert to string as a last resort
            try:
                return str(obj)
            except:
                raise TypeError(f"Object of type {type(obj).__name__} cannot be converted to a serializable type")
    
    def save(self, log_path="models_info.json"):
        """
        Save the current run to the JSON log file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        # Read existing logs
        logs = []
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as file:
                    logs = json.load(file)
            except json.JSONDecodeError:
                # If the file exists but is corrupted or empty
                logs = []
        
        # Add current run
        logs.append(self.current_run)
        
        # Save logs
        try:
            with open(log_path, 'w') as file:
                json.dump(logs, file, indent=2)
            
            print(f"Run information saved to {log_path}")
            return log_path
        except Exception as e:
            print(f"Error saving logs: {e}")
            return None