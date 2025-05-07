import matplotlib.colors as mcolors

# Define paths
OUTPUT_PATH = '../output/'
LOG_PATH = OUTPUT_PATH + 'logs/'

# Define models and configurations
MODELS = ['alexnet', 'resnet_18', 'resnet_50', 'densenet_169']
AUGMENTATION = [True, False]
DROPOUT_SETTINGS = ['fixed_0.3', 'fixed_0.5', 'custom_0.5_0.3']

# Default numeric columns for correlation matrix
DEFAULT_NUMERIC_COLS = [
    'test_accuracy', 'test_auc', 'precision_pneumonia', 'recall_pneumonia',
    'f1_pneumonia', 'false_negative_rate', 'exec_time', 'batch_size'
]

# Define base colors for models
BASE_COLORS = {
    'alexnet': 'forestgreen',    # Medium green - not too dark
    'resnet_18': 'darkorchid',   # Purple with warmth
    'resnet_50': 'tomato',       # Warm orange-red (not pure red)
    'densenet_169': 'goldenrod'  # Gold/amber color - warm without being too yellow
}

# Default palette for augmentation plot
AUG_IMPACT_PALETTE = {True: 'darkblue', False: 'lightblue'}

# Function to get light/dark shades for augmentation visualization
def get_color_shades(base_color, light_factor=1.5, dark_factor=0.7):
    """Generates lighter and darker hex color codes from a base color."""
    try:
        rgb = mcolors.to_rgb(base_color)
        hsv = mcolors.rgb_to_hsv(rgb)
        # Darker: decrease value/brightness
        dark_hsv = (hsv[0], hsv[1], max(0, min(1, hsv[2] * dark_factor)))
        # Lighter: increase value/brightness (ensure it doesn't exceed 1)
        light_hsv = (hsv[0], hsv[1] * 0.8, min(1, hsv[2] * light_factor)) # Slightly desaturate for lighter
        return mcolors.to_hex(mcolors.hsv_to_rgb(light_hsv)), mcolors.to_hex(mcolors.hsv_to_rgb(dark_hsv))
    except ValueError:
        print(f"Warning: Invalid color format '{base_color}'. Using default gray.")
        return '#cccccc', '#666666' # Default light/dark gray

# Create the final color map: maps (model_base_name, augmented_boolean) -> hex_color
AUGMENTATION_COLOR_MAP = {}
for model, color in BASE_COLORS.items():
    light_color, dark_color = get_color_shades(color)
    AUGMENTATION_COLOR_MAP[(model, False)] = light_color # Non-augmented = Lighter shade
    AUGMENTATION_COLOR_MAP[(model, True)] = dark_color  # Augmented = Darker shade