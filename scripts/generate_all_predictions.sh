#!/bin/bash

# Script to generate predictions for all models
# Processes one model at a time to avoid memory issues

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_FILE="$SCRIPT_DIR/../models.txt"
PYTHON_SCRIPT="$SCRIPT_DIR/generate_predictions.py"
DATA_PATH="../data/"
MODEL_PATH="../output/models/"
OUTPUT_PATH="../output/predictions/"

echo "=============================================="
echo "XAI Prediction Generation Script"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Models file: $MODELS_FILE"
echo "Python script: $PYTHON_SCRIPT"
echo ""

# Check if models.txt exists
if [ ! -f "$MODELS_FILE" ]; then
    echo "Error: models.txt not found at $MODELS_FILE"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Create output directory
mkdir -p "$SCRIPT_DIR/$OUTPUT_PATH"

echo "Reading models from: $MODELS_FILE"
echo ""

# Process each model
model_count=0
success_count=0
failed_models=()

while IFS= read -r line; do
    # Skip comments and empty lines
    if [[ "$line" =~ ^#.*$ ]] || [[ -z "$line" ]]; then
        continue
    fi
    
    # Remove any trailing whitespace
    model_file=$(echo "$line" | xargs)
    
    if [ -z "$model_file" ]; then
        continue
    fi
    
    echo "Processing model: $model_file"
    echo "----------------------------------------"
    
    model_count=$((model_count + 1))
    
    # Run Python script for this model
    if python3 "$PYTHON_SCRIPT" "$model_file" \
        --data_path "$DATA_PATH" \
        --model_path "$MODEL_PATH" \
        --output_path "$OUTPUT_PATH"; then
        
        echo "✓ Successfully processed: $model_file"
        success_count=$((success_count + 1))
    else
        echo "✗ Failed to process: $model_file"
        failed_models+=("$model_file")
    fi
    
    echo ""
    
    # Add a small delay to allow system cleanup
    sleep 2
    
done < "$MODELS_FILE"

echo "=============================================="
echo "PREDICTION GENERATION SUMMARY"
echo "=============================================="
echo "Total models processed: $model_count"
echo "Successful: $success_count"
echo "Failed: $((model_count - success_count))"

if [ ${#failed_models[@]} -gt 0 ]; then
    echo ""
    echo "Failed models:"
    for model in "${failed_models[@]}"; do
        echo "  - $model"
    done
fi

echo ""
echo "Predictions saved to: $SCRIPT_DIR/$OUTPUT_PATH"
echo "=============================================="

# Exit with error code if any models failed
if [ ${#failed_models[@]} -gt 0 ]; then
    exit 1
else
    echo "All models processed successfully!"
    exit 0
fi
