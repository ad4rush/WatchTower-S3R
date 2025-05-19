import os
from flask import Flask, render_template, Response, request, jsonify, current_app
import cv2
import logging
import torch
from torchvision import transforms
import numpy as np

# --- Local Imports ---
# Assuming these modules are correctly structured
try:
    from video_processor import VideoProcessor
    from ml_models.s3r_inference_wrapper import S3RInferenceWrapper
    from pytorch_resnet3d.models import resnet as i3d_resnet
except ImportError as e:
    logging.error(f"Error importing local modules: {e}")
    logging.error("Ensure the script is run from the SecurityVision root directory and module paths are correct.")
    exit() # Exit if essential modules can't be imported

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
# Number of frames per snippet for I3D
FRAMES_PER_SNIPPET = 16
# Number of snippets to feed into S3R model (Temporal dimension T)
# This should match how the S3R model was trained (e.g., 32 for UCF-Crime)
NUM_SNIPPETS_FOR_S3R = 32
# Feature dimension expected by S3R
FEATURE_DIM = 2048
# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {DEVICE}")

# --- Model Paths (Ensure these are correct relative to running app.py) ---
I3D_WEIGHTS_PATH = r"pretrained/i3d_r50_kinetics.pth"
S3R_CHECKPOINT_PATH = r"checkpoint/ucf-crime_s3r_i3d_best.pth"
S3R_DICTIONARY_PATH = r"dictionary/ucf-crime/ucf-crime_dictionaries.taskaware.omp.100iters.50pct.npy"

# --- I3D Preprocessing Transform ---
i3d_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Global Variables ---
# Initialize models and processor as None initially
i3d_model = None
s3r_wrapper = None
video_processor = None
camera = None # Global camera object

# --- Model Loading Functions ---
def load_i3d_model(weights_path):
    """Loads the I3D model and weights."""
    logging.info("Loading I3D model...")
    try:
        net = i3d_resnet.i3_res50(num_classes=400)
        abs_weights_path = os.path.abspath(weights_path)
        if not os.path.exists(weights_path):
            logging.error(f"I3D weights file not found at '{abs_weights_path}'")
            return None
        weights = torch.load(weights_path, map_location=DEVICE)
        if "state_dict" in weights:
            state_dict = {k.replace('module.', ''): v for k, v in weights['state_dict'].items()}
        else:
            state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
        net.load_state_dict(state_dict)
        net.to(DEVICE)
        net.eval()
        logging.info("I3D Model loaded successfully.")
        return net
    except Exception as e:
        logging.error(f"Error loading I3D model: {e}", exc_info=True)
        return None

def load_s3r_model(checkpoint_path, dictionary_path):
    """Loads the S3R model using the inference wrapper."""
    logging.info("Loading S3R model...")
    abs_checkpoint_path = os.path.abspath(checkpoint_path)
    abs_dictionary_path = os.path.abspath(dictionary_path)
    if not os.path.exists(checkpoint_path):
        logging.error(f"S3R checkpoint file not found at '{abs_checkpoint_path}'")
        return None
    if not os.path.exists(dictionary_path):
        logging.error(f"S3R dictionary file not found at '{abs_dictionary_path}'")
        return None
    try:
        wrapper = S3RInferenceWrapper(
            checkpoint_path=checkpoint_path,
            dictionary_path=dictionary_path,
            device=DEVICE,
            feature_dim=FEATURE_DIM # Should be 2048 for I3D
        )
        logging.info("S3R Model loaded successfully via wrapper.")
        return wrapper
    except Exception as e:
        logging.error(f"Error loading S3R model via wrapper: {e}", exc_info=True)
        return None

# --- Flask App Initialization ---
def create_app():
    """Creates and configures the Flask application."""
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.config['SECRET_KEY'] = 'your_secret_key' # Change this for production

    global i3d_model, s3r_wrapper, video_processor

    # Load models during app creation
    logging.info("Loading machine learning models...")
    i3d_model = load_i3d_model(I3D_WEIGHTS_PATH)
    s3r_wrapper = load_s3r_model(S3R_CHECKPOINT_PATH, S3R_DICTIONARY_PATH)

    if i3d_model is None or s3r_wrapper is None:
        logging.error("Failed to load one or more models. Application cannot start.")
        # Depending on desired behavior, you might exit or run with limited functionality
        exit() # Exit if models are essential

    # Initialize Video Processor with the loaded I3D model
    video_processor = VideoProcessor(
        i3d_model=i3d_model,
        i3d_transform=i3d_transform,
        device=DEVICE,
        frames_per_snippet=FRAMES_PER_SNIPPET,
        num_snippets_for_s3r=NUM_SNIPPETS_FOR_S3R
    )
    logging.info("VideoProcessor initialized.")

    # Register blueprints/routes
    # Assuming routes are defined in routes.py and imported
    try:
        from routes import main_bp
        app.register_blueprint(main_bp)
        logging.info("Registered routes blueprint.")
    except ImportError:
        logging.error("Could not import routes blueprint. Ensure routes.py exists and is importable.")
    except Exception as e:
         logging.error(f"Error registering blueprint: {e}", exc_info=True)

    # --- Pass models and processor to routes using app context ---
    # This makes them accessible within your route functions via current_app
    app.config['I3D_MODEL'] = i3d_model
    app.config['S3R_WRAPPER'] = s3r_wrapper
    app.config['VIDEO_PROCESSOR'] = video_processor
    app.config['CAMERA'] = camera # Store camera globally if needed across requests

    return app

# --- Main Execution ---
if __name__ == '__main__':
    app = create_app()
    # Consider using Waitress or Gunicorn for production instead of Flask's dev server
    app.run(debug=True, host='0.0.0.0', port=5000) # Runs on http://localhost:5000
