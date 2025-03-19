import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(history):
    """Plot training history with accuracy, loss and ROC AUC"""
    # Determine how many metrics to plot
    has_auc = any(metric for metric in history.history.keys() if 'auc' in metric.lower())
    num_plots = 3 if has_auc else 2
    
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    # ROC AUC plot if available
    if has_auc:
        auc_metric = next(metric for metric in history.history.keys() if 'auc' in metric.lower() and not metric.startswith('val'))
        val_auc_metric = f'val_{auc_metric}'
        
        axes[2].plot(history.history[auc_metric], label=f'Training AUC')
        axes[2].plot(history.history[val_auc_metric], label=f'Validation AUC')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Training and Validation AUC')
        axes[2].legend()
        axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.6f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
    
    print(f"ROC AUC Score: {roc_auc:.4f}")

def plot_confusion_matrix(cm):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_model_comparison(results1, results2, title='Model Comparison'):
    """Plot comparison of two models"""
    labels = ['Accuracy', 'AUC']
    model1_values = [results1['test_accuracy'], results1['roc']['auc']]
    model2_values = [results2['test_accuracy'], results2['roc']['auc']]
    
    x = range(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x, model1_values, width, label='Best Model')
    ax.bar([i + width for i in x], model2_values, width, label='Selected Model')
    
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks([i + width/2 for i in x])
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.show()