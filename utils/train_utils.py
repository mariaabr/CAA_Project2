from tensorflow.keras import callbacks

def get_callbacks(model_path, monitor='val_AUC', mode='max', patience=10, factor=0.2):
    """Get standard callbacks for model training"""
    checkpoint_callback = callbacks.ModelCheckpoint(
        model_path,
        save_best_only=True,
        monitor=monitor,
        mode=mode
    )
    
    early_stop_callback = callbacks.EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        restore_best_weights=True
    )
    
    reduce_lr_callback = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=factor,
        patience=patience//2
    )
    
    all_callbacks = [
        checkpoint_callback,
        early_stop_callback,
        reduce_lr_callback
    ]
    
    return all_callbacks

def train_model(model, train_data, train_labels, val_data, val_labels, 
                epochs=100, batch_size=32, callbacks=None, verbose=1):
    """Train a model with given parameters and callbacks"""
    history = model.fit(
        train_data,
        train_labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_data, val_labels),
        callbacks=callbacks,
        verbose=verbose
    )
    return history