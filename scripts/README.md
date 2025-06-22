# XAI Analysis Scripts

This directory contains scripts for generating model predictions efficiently, avoiding memory issues when processing multiple large models.

## Files

### `generate_predictions.py`
Standalone Python script that processes a single model to generate predictions on test data.

**Usage:**
```bash
python3 generate_predictions.py <model_filename> [options]
```

**Arguments:**
- `model_filename`: Name of the model file (e.g., `alexnet_augmented_b96_lr0.001_dr0.5_pneumonia_model.keras`)

**Options:**
- `--data_path`: Path to data directory (default: `../data/`)
- `--model_path`: Path to models directory (default: `../output/models/`)
- `--output_path`: Path to save predictions (default: `../output/predictions/`)

**Example:**
```bash
python3 generate_predictions.py alexnet_augmented_b96_lr0.001_dr0.5_pneumonia_model.keras
```

### `generate_all_predictions.sh`
Bash script that processes all models listed in `../models.txt` using the Python script above.

**Usage:**
```bash
./generate_all_predictions.sh
```

**Features:**
- Processes models one at a time to avoid memory issues
- Skips models if predictions already exist
- Provides detailed progress and error reporting
- Automatic memory cleanup between models

## Workflow

1. **Make the bash script executable:**
   ```bash
   chmod +x generate_all_predictions.sh
   ```

2. **Generate all predictions:**
   ```bash
   ./generate_all_predictions.sh
   ```

3. **Use predictions in the XAI notebook:**
   - Set `SKIP_PREDICTION_GENERATION = True` in the notebook configuration
   - Run the XAI analysis notebook

## Output

Predictions are saved as pickle files in `../output/predictions/` with the naming pattern:
`{architecture}_{resolution}px_predictions.pkl`

Each prediction file contains:
- Model information (architecture, resolution, etc.)
- Predicted classes and probabilities
- True labels
- Full prediction scores
- Accuracy metrics
- Sample indices

## Memory Management

The scripts are designed to:
- Load only one model at a time
- Clear TensorFlow session between models
- Use garbage collection to free memory
- Process models sequentially to avoid conflicts

## Troubleshooting

**If you get import errors:**
- Ensure you're running from the scripts directory
- Check that the parent directory structure is correct
- Verify that TensorFlow and other dependencies are installed

**If models are not found:**
- Check that model files exist in `../output/models/`
- Verify model filenames in `../models.txt`
- Ensure paths are correct relative to the scripts directory

**For memory issues:**
- Close other applications
- Use a machine with more RAM
- Process models individually rather than in batch
