import pandas as pd
import scipy.stats as stats

from utils.results.constants import MODELS, BASE_COLORS, AUGMENTATION_COLOR_MAP, AUG_IMPACT_PALETTE, DROPOUT_SETTINGS, DEFAULT_NUMERIC_COLS

def get_best_model_executions(df, metrics=None):
    """
    Creates a table showing the best execution of each model architecture ranked by AUC.
    
    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results
        metrics (list): Optional list of additional metrics to include in the table
    
    Returns:
        pd.DataFrame: Table of best models ranked by AUC
    """
    if df.empty:
        print("Cannot create table. DataFrame is empty.")
        return pd.DataFrame()
    
    # Ensure required columns exist
    if 'model_base_name' not in df.columns or 'test_auc' not in df.columns:
        print("Required columns 'model_base_name' or 'test_auc' missing from DataFrame")
        return pd.DataFrame()
    
    # Default metrics to include
    if metrics is None:
        metrics = ['test_auc', 'test_accuracy', 'false_negative_rate', 'fn']
    
    # Find the best execution for each model architecture
    best_models = []
    
    for model_name in df['model_base_name'].unique():
        model_df = df[df['model_base_name'] == model_name].dropna(subset=['test_auc'])
        
        if not model_df.empty:
            # Get the row with the highest AUC
            best_idx = model_df['test_auc'].idxmax()
            best_row = model_df.loc[best_idx].to_dict()
            best_models.append(best_row)
    
    if not best_models:
        print("No valid model executions found with AUC values.")
        return pd.DataFrame()
    
    # Create DataFrame and sort by AUC descending
    result_df = pd.DataFrame(best_models)
    result_df = result_df.sort_values('test_auc', ascending=False).reset_index(drop=True)
    
    # Add rank column
    result_df.insert(0, 'Rank', range(1, len(result_df) + 1))
    
    # Select relevant columns that exist in the data
    display_cols = ['Rank', 'model_base_name', 'augmentation']
    for metric in metrics:
        if metric in result_df.columns:
            display_cols.append(metric)
    
    # Format columns for display
    result_df = result_df[display_cols].copy()
    
    # Convert augmentation to Yes/No
    if 'augmentation' in result_df.columns:
        result_df['augmentation'] = result_df['augmentation'].map({True: 'Yes', False: 'No'})
    
    # Format percentages and numbers
    if 'false_negative_rate' in result_df.columns:
        result_df['false_negative_rate'] = result_df['false_negative_rate'].map('{:.2%}'.format)
    if 'test_auc' in result_df.columns:
        result_df['test_auc'] = result_df['test_auc'].map('{:.4f}'.format)
    if 'test_accuracy' in result_df.columns:
        result_df['test_accuracy'] = result_df['test_accuracy'].map('{:.4f}'.format)
    
    # Rename columns for clarity
    column_rename_map = {
        'model_base_name': 'Model',
        'augmentation': 'Augmentation',
        'test_auc': 'AUC',
        'test_accuracy': 'Accuracy',
        'false_negative_rate': 'FNR',
        'fn': 'FN Count'
    }

    # Set index to rank
    result_df.set_index('Rank', inplace=True)
    
    return result_df.rename(columns=column_rename_map)

def get_metric_comparison_data(df, metric='test_auc', title_metric='AUC'):
    """
    Creates a dataframe with metric comparison data by dropout strategy, model, and augmentation.
    
    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results
        metric (str): The metric column to analyze
        title_metric (str): The formatted name of the metric for display
    
    Returns:
        pd.DataFrame: Table with mean values and confidence intervals
    """
    # Create custom order for the hue based on the MODELS constant
    custom_hue_order = []
    for model in MODELS:  
        custom_hue_order.append((model, False))  # No Aug
        custom_hue_order.append((model, True))   # Aug
    
    # Filter to only include combinations that exist in the data
    custom_hue_order = [combo for combo in custom_hue_order 
                       if combo in df['hue_combined'].unique()]
    
    # Create results dataframe with all bar data
    results = []
    
    # Iterate through dropout settings
    for dropout in DROPOUT_SETTINGS:
        # Get data for this dropout setting
        dropout_data = df[df['dropout_setting'] == dropout]
        
        # For each model/augmentation combination
        for model, aug in custom_hue_order:
            model_aug_data = dropout_data[
                (dropout_data['model_base_name'] == model) & 
                (dropout_data['augmentation'] == aug)
            ]
            
            # Only include if we have data
            if not model_aug_data.empty:
                # Calculate statistics
                mean_value = model_aug_data[metric].mean()
                
                # Calculate 95% confidence interval
                n = len(model_aug_data)
                if n > 1:  # Can only calculate CI with more than one sample
                    sem = model_aug_data[metric].sem()
                    ci_95 = stats.t.interval(0.95, n-1, loc=mean_value, scale=sem)
                    lower_ci, upper_ci = ci_95
                else:
                    # If only one sample, use the value itself
                    lower_ci = upper_ci = mean_value
                
                results.append({
                    'Dropout Strategy': dropout,
                    'Model': model,
                    'Augmentation': aug,
                    f'Mean {title_metric}': mean_value,
                    f'Lower CI (95%)': lower_ci,
                    f'Upper CI (95%)': upper_ci
                })
    
    # Create dataframe from collected results
    return pd.DataFrame(results)

def get_false_negative_comparison_data(df):
    """
    Creates a dataframe with false negative rate comparisons across models and augmentation.
    
    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results
    
    Returns:
        pd.DataFrame: Table with false negative statistics
    """
    plot_df = df[df['false_negative_rate'].notna()].copy()
    if plot_df.empty:
        print("No data available for False Negative Rate analysis.")
        return pd.DataFrame()
    
    # Create results dataframe with mean and extreme values
    results = []
    
    # For each model/augmentation combination
    for model in MODELS:
        for aug in [False, True]:
            model_aug_data = plot_df[
                (plot_df['model_base_name'] == model) & 
                (plot_df['augmentation'] == aug)
            ]
            
            # Only include if we have data
            if not model_aug_data.empty:
                # Calculate statistics
                mean_value = model_aug_data['false_negative_rate'].mean()
                min_value = model_aug_data['false_negative_rate'].min()  # Best case (lowest FNR)
                max_value = model_aug_data['false_negative_rate'].max()  # Worst case (highest FNR)
                
                # Calculate 95% confidence interval
                n = len(model_aug_data)
                if n > 1:  # Can only calculate CI with more than one sample
                    sem = model_aug_data['false_negative_rate'].sem()
                    ci_95 = stats.t.interval(0.95, n-1, loc=mean_value, scale=sem)
                    lower_ci, upper_ci = ci_95
                else:
                    # If only one sample, use the value itself
                    lower_ci = upper_ci = mean_value
                
                results.append({
                    'Model': model,
                    'Augmentation': aug,
                    'Mean False Negative Rate': mean_value,
                    'Best Case (Min FNR)': min_value,
                    'Worst Case (Max FNR)': max_value,
                    'Lower CI (95%)': lower_ci,
                    'Upper CI (95%)': upper_ci,
                    'Sample Size': n
                })
    
    # Create dataframe from collected results
    return pd.DataFrame(results)

def get_execution_time_comparison_data(df):
    """
    Creates a dataframe with execution time comparisons across models, batch sizes, and augmentation.
    
    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results
    
    Returns:
        pd.DataFrame: Table with execution time statistics
    """
    plot_df = df[df['exec_time'].notna()].copy()
    if plot_df.empty:
        print("No data available for Execution Time analysis.")
        return pd.DataFrame()
    
    # Create results dataframe with all data combinations
    results = []
    
    # For each model/augmentation/batch_size combination
    for model in MODELS:
        for aug in [False, True]:
            for batch_size in plot_df['batch_size'].unique():
                subset = plot_df[
                    (plot_df['model_base_name'] == model) & 
                    (plot_df['augmentation'] == aug) &
                    (plot_df['batch_size'] == batch_size)
                ]
                
                # Only include if we have data
                if not subset.empty:
                    # Calculate statistics
                    mean_value = subset['exec_time'].mean()
                    min_value = subset['exec_time'].min()  # Best case (fastest time)
                    max_value = subset['exec_time'].max()  # Worst case (slowest time)
                    
                    # Calculate 95% confidence interval
                    n = len(subset)
                    if n > 1:  # Can only calculate CI with more than one sample
                        sem = subset['exec_time'].sem()
                        ci_95 = stats.t.interval(0.95, n-1, loc=mean_value, scale=sem)
                        lower_ci, upper_ci = ci_95
                    else:
                        # If only one sample, use the value itself
                        lower_ci = upper_ci = mean_value
                    
                    results.append({
                        'Model': model,
                        'Augmentation': aug,
                        'Batch Size': batch_size,
                        'Mean Execution Time (s)': mean_value,
                        'Best Case (Min Time)': min_value,
                        'Worst Case (Max Time)': max_value,
                        'Lower CI (95%)': lower_ci,
                        'Upper CI (95%)': upper_ci,
                        'Sample Size': n
                    })
    
    # Create dataframe from collected results
    return pd.DataFrame(results)

def get_augmentation_impact_data(df, metric='test_auc'):
    """
    Creates a dataframe showing the impact of augmentation on performance metrics by model type.
    
    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results
        metric (str): The metric to analyze
        
    Returns:
        pd.DataFrame: Table with augmentation impact data
    """
    # Check if data is available
    if df.empty or metric not in df.columns:
        print(f"No data available for {metric} augmentation impact analysis.")
        return pd.DataFrame()
    
    # Aggregate mean metric by model type and augmentation status
    aug_by_model = (
        df.groupby(['model_base_name', 'augmentation'], observed=False)[metric]
        .agg(['mean', 'count', 'std'])
        .reset_index()
    )
    
    # Calculate difference between augmented and non-augmented
    impact_data = []
    
    for model in MODELS:
        model_data = aug_by_model[aug_by_model['model_base_name'] == model]
        
        if len(model_data) == 2:  # Both augmented and non-augmented exist
            aug_value = model_data[model_data['augmentation'] == True]['mean'].values[0]
            no_aug_value = model_data[model_data['augmentation'] == False]['mean'].values[0]
            
            impact_data.append({
                'Model': model,
                f'{metric} (Augmented)': aug_value,
                f'{metric} (No Aug)': no_aug_value,
                'Difference': aug_value - no_aug_value,
                'Percent Improvement': (aug_value - no_aug_value) / no_aug_value * 100
            })
    
    return pd.DataFrame(impact_data)

def create_best_models_table(df, top_n=10, sort_by=None):
    """
    Create a formatted table of the best performing models with multi-metric sorting.
    
    Args:
        df (pd.DataFrame): DataFrame containing model data
        top_n (int): Number of top models to display
        sort_by (list or str): List of columns to sort by, with each item being either:
                            - A string column name (will use ascending=False by default)
                            - A tuple of (column_name, ascending_bool)
                            If a string is provided, it will be converted to a single-item list
    
    Returns:
        pd.DataFrame: Formatted table of top models
    """
    if df.empty:
        print("Cannot create table. DataFrame is empty.")
        return pd.DataFrame()
    
    # Handle sort_by parameter
    if sort_by is None:
        sort_by = [('test_auc', False)]  # Default sort by test_auc descending
    elif isinstance(sort_by, str):
        # For single column string, convert to list with default ascending=False
        # Special case: false_negative_rate should be ascending=True (lower is better)
        ascending = True if sort_by == 'false_negative_rate' else False
        sort_by = [(sort_by, ascending)]
    elif isinstance(sort_by, list):
        # Process each item in the list
        processed_sort_by = []
        for item in sort_by:
            if isinstance(item, str):
                # If string, use default ascending value based on metric name
                ascending = True if item == 'false_negative_rate' else False
                processed_sort_by.append((item, ascending))
            elif isinstance(item, tuple) and len(item) == 2:
                # If tuple with column and ascending boolean, use as is
                processed_sort_by.append(item)
            else:
                print(f"Warning: Invalid sort specification {item}. Skipping.")
        sort_by = processed_sort_by
    
    # Check if all sort columns exist
    sort_columns = [col for col, _ in sort_by]
    missing_cols = [col for col in sort_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Sort columns {missing_cols} not found in DataFrame. Using available columns only.")
        sort_by = [(col, asc) for col, asc in sort_by if col in df.columns]
        if not sort_by:
            print("No valid sort columns. Using default sort by test_auc.")
            sort_by = [('test_auc', False)]
    
    # Select and copy relevant columns
    cols = ['model_base_name', 'augmentation', 'batch_size', 'dropout_setting',
            'test_accuracy', 'test_auc', 'precision_pneumonia', 'recall_pneumonia',
            'f1_pneumonia', 'false_negative_rate', 'fn', 'exec_time']
    
    # Make sure to include all sort columns
    for col, _ in sort_by:
        if col not in cols:
            cols.append(col)
    
    # Filter to columns that exist in the dataframe
    cols = [col for col in cols if col in df.columns]
    
    # Create a copy of the data with only needed columns
    table_df = df[cols].copy()
    
    # Remove rows with NaN in any sort column
    for col, _ in sort_by:
        table_df = table_df.dropna(subset=[col])
    
    if table_df.empty:
        print("All rows contain NaN values in sort columns.")
        return pd.DataFrame()
    
    # Sort the data
    sort_cols = [col for col, _ in sort_by]
    sort_ascending = [asc for _, asc in sort_by]
    table_df = table_df.sort_values(sort_cols, ascending=sort_ascending)
    
    # Format columns for display
    format_map = {
        'test_accuracy': '{:.8f}', 
        'test_auc': '{:.8f}',
        'precision_pneumonia': '{:.4f}', 
        'precision': '{:.4f}',
        'recall_pneumonia': '{:.4f}',
        'true_positive_rate': '{:.4f}',
        'false_positive_rate': '{:.4%}',
        'f1_pneumonia': '{:.4f}',
        'false_negative_rate': '{:.4%}',
        'fn': '{:.0f}',  # Raw false negatives
        'exec_time': '{:.2f}s'
    }
    
    formatted_df = table_df.copy()
    for col, fmt in format_map.items():
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].map(lambda x: fmt.format(x) if pd.notna(x) else 'N/A')
    
    # Rename columns for better display
    column_name_map = {
        'model_base_name': 'Model',
        'augmentation': 'Augmentation',
        'batch_size': 'Batch Size',
        'dropout_setting': 'Dropout Strategy',
        'test_accuracy': 'Accuracy',
        'test_auc': 'AUC',
        'precision_pneumonia': 'Precision (Pneu)',
        'precision': 'Precision',
        'recall_pneumonia': 'Recall (Pneu)',
        'true_positive_rate': 'TPR',
        'false_positive_rate': 'FPR',
        'f1_pneumonia': 'F1 (Pneu)',
        'false_negative_rate': 'FN Rate',
        'fn': 'FN Count',
        'exec_time': 'Execution Time'
    }
    
    formatted_df = formatted_df.rename(columns={col: column_name_map.get(col, col) 
                                                for col in formatted_df.columns})
    
    return formatted_df.head(top_n)
