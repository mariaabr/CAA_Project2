import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def evaluate_model(model, test_images, test_labels, verbose=1):
    """Evaluate model and return metrics"""
    # Evaluate the model
    test_loss, test_accuracy, test_auc = model.evaluate(test_images, test_labels, verbose=verbose)
    
    # Make predictions (with no progress bar)
    predictions = model.predict(test_images, verbose=verbose)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels[:, 1], predictions[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    
    # Classification report
    report = classification_report(true_classes, predicted_classes, 
                                    target_names=['Normal', 'Pneumonia'])
    
    results = {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_auc': test_auc,
        'predictions': predictions,
        'predicted_classes': predicted_classes,
        'true_classes': true_classes,
        'roc': {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc},
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    return results
