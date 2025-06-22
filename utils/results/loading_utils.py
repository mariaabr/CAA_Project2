import os
import json
import pandas as pd

from utils.results.constants import LOG_PATH, MODELS, AUGMENTATION_COLOR_MAP

def load_model_logs():
    """
    Load all model log files from the logs directory
    
    Returns:
        dict: Dictionary mapping model files to their JSON contents
    """
    log_files = {}
    
    for filename in os.listdir(LOG_PATH):
        if filename.endswith('_info.json'):
            file_path = os.path.join(LOG_PATH, filename)
            try:
                with open(file_path, 'r') as file:
                    log_files[filename] = json.load(file)
                    print(f"Loaded {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return log_files

def extract_model_data(log_files):
    """
    Extract and structure data from model logs into a Pandas DataFrame.

    Args:
        log_files (dict): Dictionary mapping model filenames to their JSON contents.

    Returns:
        pd.DataFrame: DataFrame containing structured model data.
    """
    all_data = []

    for filename, logs in log_files.items():
        # Extract model information from filename
        model_info = extract_model_info_from_filename(filename)
        if not model_info:
            continue
        
        model_base_name, is_augmented_from_filename = model_info
        
        # Normalize logs to a list
        if not isinstance(logs, list):
            logs = [logs]

        # Process each run in the logs
        for run in logs:
            if not isinstance(run, dict):
                print(f"Warning: Skipping invalid run data in {filename}.")
                continue
            
            # Create run info dictionary with all extracted data
            run_info = extract_run_data(filename, model_base_name, is_augmented_from_filename, run)
            all_data.append(run_info)

    if not all_data:
        print("Warning: No valid data extracted from log files.")
        return pd.DataFrame()

    # Convert to DataFrame and process
    return finalize_dataframe(all_data)


####################################################################
## Functions related to extracting the model data from json files ##
####################################################################


def extract_model_info_from_filename(filename):
    """Extract the model base name and augmentation status from filename."""
    base_filename = filename.replace('_info.json', '')
    is_augmented_from_filename = '_augmented' in base_filename

    # Find the base model name (longest match first if overlapping names exist)
    matched_model = None
    for m in sorted(MODELS, key=len, reverse=True):
        if m in base_filename:
            matched_model = m
            break

    if not matched_model:
        print(f"Warning: Could not determine model type for {filename}. Skipping.")
        return None
    
    return matched_model, is_augmented_from_filename


def extract_run_data(filename, model_base_name, is_augmented_from_filename, run):
    """Extract all relevant data for a single run."""
    # Get augmentation status
    run_augmented = run.get('augmentation', is_augmented_from_filename)
    if isinstance(run_augmented, str):
        run_augmented = run_augmented.lower() == 'true' or run_augmented == 'augmentation'
    
    # Create basic info dictionary
    run_info = {
        'filename': filename,
        'model_base_name': model_base_name,
        'augmentation': bool(run_augmented),
        'timestamp': run.get('timestamp'),
        'batch_size': run.get('batch_size'),
        'learning_rate': run.get('learning_rate'),
        'dropout_mode': run.get('dropout_mode'),
        'exec_time': run.get('exec_time')
    }
    
    # Add combined hue column for plotting
    run_info['hue_combined'] = (run_info['model_base_name'], run_info['augmentation'])
    
    # Add dropout information
    add_dropout_info(run_info, run)
    
    # Add evaluation metrics
    add_evaluation_metrics(run_info, run)
    
    # Add training history data
    add_training_history(run_info, run)
    
    return run_info


def add_dropout_info(run_info, run):
    """Add dropout configuration information to run_info."""
    dropout_mode = run_info['dropout_mode']
    
    if dropout_mode == 'fixed':
        rate = run.get('dropout_rate')
        run_info['dropout_rate'] = rate
        run_info['dropout_setting'] = f"fixed_{rate}" if rate is not None else "fixed_unknown"
    elif dropout_mode == 'custom':
        start = run.get('dropout_start')
        end = run.get('dropout_end')
        run_info['dropout_start'] = start
        run_info['dropout_end'] = end
        run_info['dropout_setting'] = f"custom_{start}_{end}" if start is not None and end is not None else "custom_unknown"
    else:
        run_info['dropout_setting'] = 'unknown'


def add_evaluation_metrics(run_info, run):
    """Add evaluation metrics to run_info."""
    eval_data = run.get('evaluation', {})
    
    # Add basic metrics
    run_info['test_accuracy'] = eval_data.get('test_accuracy')
    run_info['test_loss'] = eval_data.get('test_loss')
    
    # Use the ROC AUC calculated from fpr/tpr if available, else the direct test_auc
    run_info['test_auc'] = eval_data.get('roc', {}).get('auc', eval_data.get('test_auc'))
    
    # Extract confusion matrix values if available
    cm = eval_data.get('confusion_matrix')
    if isinstance(cm, list) and len(cm) == 2 and all(isinstance(row, list) and len(row) == 2 for row in cm):
        try:
            tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])
            run_info.update({
                'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
            })
            
            # Calculate metrics directly from confusion matrix
            # True positives + False negatives = Total actual positives
            total_actual_pos = tp + fn
            
            # True negatives + False positives = Total actual negatives
            total_actual_neg = tn + fp
            
            # Calculate rates directly from confusion matrix
            if total_actual_pos > 0:
                run_info['true_positive_rate'] = tp / total_actual_pos  # TPR = Recall = Sensitivity
                run_info['false_negative_rate'] = fn / total_actual_pos  # FNR = 1 - TPR
            
            if total_actual_neg > 0:
                run_info['true_negative_rate'] = tn / total_actual_neg  # TNR = Specificity
                run_info['false_positive_rate'] = fp / total_actual_neg  # FPR = 1 - TNR
            
            # Calculate precision if possible
            predicted_pos = tp + fp
            if predicted_pos > 0:
                run_info['precision'] = tp / predicted_pos
            
            # Extract metrics from classification report if available for comparison
            class_report = eval_data.get('classification_report', {})
            classes = class_report.get('classes', {})
            averages = class_report.get('averages', {})
            
            # Get pneumonia metrics from classification report
            pneumonia_class = classes.get('Pneumonia', {})
            run_info['precision_pneumonia'] = pneumonia_class.get('precision')
            run_info['recall_pneumonia'] = pneumonia_class.get('recall')
            run_info['f1_pneumonia'] = pneumonia_class.get('f1-score')
            
            # Use weighted average as the main metrics (more representative for imbalanced dataset)
            weighted_avg = averages.get('weighted avg', {})
            run_info['precision'] = weighted_avg.get('precision')
            run_info['recall'] = weighted_avg.get('recall')
            run_info['f1_score'] = weighted_avg.get('f1-score')
            
            # Also store macro average as backup
            macro_avg = averages.get('macro avg', {})
            run_info['precision_macro'] = macro_avg.get('precision')
            run_info['recall_macro'] = macro_avg.get('recall')
            run_info['f1_score_macro'] = macro_avg.get('f1-score')
            
        except (TypeError, ValueError) as e:
            print(f"Warning: Error processing confusion matrix in {run_info['filename']}: {cm}. Error: {e}")


def add_training_history(run_info, run):
    """Add training history data to run_info."""
    history = run.get('training_history', {})
    
    # Add final values for important metrics
    for metric in ['val_loss', 'val_accuracy', 'val_AUC']:
        values = history.get(metric)
        if isinstance(values, list) and values:
            run_info[f'final_{metric}'] = values[-1]


def finalize_dataframe(all_data):
    """Convert data to DataFrame and finalize processing."""
    df = pd.DataFrame(all_data)
    
    # Define numeric columns
    numeric_cols = [
        'batch_size', 'learning_rate', 'dropout_rate', 'dropout_start', 'dropout_end',
        'exec_time', 'test_accuracy', 'test_auc', 'test_loss',
        'tn', 'fp', 'fn', 'tp', 
        'precision', 'recall', 'f1_score',
        'precision_pneumonia', 'recall_pneumonia', 'f1_pneumonia',
        'precision_macro', 'recall_macro', 'f1_score_macro',
        'true_positive_rate', 'false_negative_rate',
        'true_negative_rate', 'false_positive_rate'
    ]
    numeric_cols.extend([f'final_{metric}' for metric in ['val_loss', 'val_accuracy', 'val_AUC']])
    
    # Convert only columns that exist in the dataframe
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce errors to NaN

    # Apply categorical ordering
    if 'model_base_name' in df.columns:
        df['model_base_name'] = pd.Categorical(df['model_base_name'], categories=MODELS, ordered=True)
    
    if 'dropout_setting' in df.columns:
        df['dropout_setting'] = pd.Categorical(df['dropout_setting'])
    
    # Make combined hue categorical
    if 'hue_combined' in df.columns:
        df['hue_combined'] = pd.Categorical(
            df['hue_combined'], 
            categories=sorted(AUGMENTATION_COLOR_MAP.keys()), 
            ordered=True
        )
    
    # Sort for consistency
    sort_columns = [col for col in ['model_base_name', 'augmentation', 'batch_size', 'dropout_setting'] 
                if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns)

    # Log completion and check for missing data
    print(f"Data extraction complete. {len(df)} runs processed.")
    
    # Check for missing crucial data
    for critical_metric in ['test_auc', 'false_negative_rate', 'false_positive_rate']:
        if critical_metric in df.columns:
            missing = df[critical_metric].isnull().sum()
            if missing > 0:
                print(f"Warning: {missing} runs have missing '{critical_metric}' values.")

    return df