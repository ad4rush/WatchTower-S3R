import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import sys
from numpy.linalg import norm # For calculating vector magnitude (norm)

# --- Configuration ---

# >>> IMPORTANT: Set the paths for your two images <<<
IMAGE_PATH_1 = r"s.png" # <-- First image path
IMAGE_PATH_2 = r"random.jpeg" # <-- Second image path

# --- Relative Paths (assuming script is run from SecurityVision directory) ---

# Path to I3D weights (verify this exists)
I3D_WEIGHTS_PATH = r"pretrained/i3d_r50_kinetics.pth"

# S3R paths are not needed for this script
# S3R_CHECKPOINT_PATH = r"checkpoint/ucf-crime_s3r_i3d_best.pth"
# S3R_DICTIONARY_PATH = r"dictionary/ucf-crime/ucf-crime_dictionaries.taskaware.omp.100iters.50pct.npy"

# --- Python Path Setup ---
# If needed, uncomment and adjust:
# sys.path.insert(0, os.path.abspath('.'))

# --- Model Setup ---
try:
    # Imports should work if run from the SecurityVision directory
    from pytorch_resnet3d.models import resnet as i3d_resnet
    # S3RInferenceWrapper is not needed for this script
    # from ml_models.s3r_inference_wrapper import S3RInferenceWrapper
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure you are running this script from the main 'SecurityVision' directory,")
    print("and that the 'pytorch_resnet3d' subdirectory exists.")
    sys.exit(1)

# Constants for I3D
FRAMES_PER_CLIP = 16 # Number of times to replicate image for I3D input clip
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Preprocessing ---
# Standard I3D preprocessing
i3d_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)), # Resize to I3D input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet normalization
])

# --- Feature Extraction Function ---
def extract_features_from_image(image_path, model, transform, features_list):
    """
    Loads a single image, replicates it FRAMES_PER_CLIP times to form a clip,
    and extracts I3D features.

    Args:
        image_path (str): Path to the input image.
        model (torch.nn.Module): The loaded I3D model.
        transform (transforms.Compose): Preprocessing transform for I3D.
        features_list (list): A list to store the extracted features via hook.

    Returns:
        np.ndarray or None: The extracted feature vector (2048 dimensions) or None if error.
    """
    print(f"\nProcessing image: {image_path}")
    # Load the image
    abs_image_path = os.path.abspath(image_path)
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{abs_image_path}'")
        return None
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image file '{abs_image_path}'")
        return None

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the single frame
    processed_frame = transform(frame) # Shape: [3, 224, 224]

    # Replicate the frame to create a clip for I3D
    # I3D expects input shape [Batch, Channels, Time, Height, Width]
    clip = torch.stack([processed_frame] * FRAMES_PER_CLIP, dim=1) # Shape: [3, T=16, 224, 224]
    clip = clip.unsqueeze(0).to(DEVICE) # Shape: [1, 3, T=16, 224, 224]

    print(f"Input clip shape for I3D: {clip.shape}")

    # Extract features using the hook on the avgpool layer
    with torch.no_grad():
        features_list.clear() # Clear list before inference
        _ = model({'frames': clip}) # Pass the clip through the model

        if not features_list:
            print("Error: I3D feature extraction hook did not capture any output.")
            return None

        # Get the feature captured by the hook
        feature = features_list[0] # Output from avgpool layer
        # Squeeze removes unnecessary dimensions (like batch), convert to numpy
        feature = feature.squeeze().cpu().numpy()

    # Expected feature shape is (2048,)
    print(f"Extracted features shape: {feature.shape}")
    # Optionally print first few features
    # print(f"First 10 features: {feature[:10]}")
    return feature

# --- Model Loading Function ---
def load_i3d_model(weights_path):
    """Loads the I3D model and weights."""
    print("Loading I3D model...")
    try:
        # Initialize I3D model structure (ResNet50 backbone for Kinetics)
        net = i3d_resnet.i3_res50(num_classes=400)

        # Check if weights file exists
        abs_weights_path = os.path.abspath(weights_path)
        if not os.path.exists(weights_path):
            print(f"Error: I3D weights file not found at '{abs_weights_path}'")
            print("Please ensure the path is correct relative to the SecurityVision directory.")
            return None

        # Load the weights
        weights = torch.load(weights_path, map_location=DEVICE)

        # Handle different checkpoint formats (common practice)
        if "state_dict" in weights:
            state_dict = {k.replace('module.', ''): v for k, v in weights['state_dict'].items()}
            net.load_state_dict(state_dict)
        else:
            state_dict = {k.replace('module.', ''): v for k, v in weights.items()}
            net.load_state_dict(state_dict)

        net.to(DEVICE)
        net.eval()
        print("I3D Model loaded successfully.")
        return net
    except Exception as e:
        print(f"Error loading I3D model: {e}")
        return None

# --- Feature Comparison Function ---
def compare_features(feat1, feat2):
    """Calculates Euclidean distance and Cosine similarity between two feature vectors."""
    if feat1 is None or feat2 is None:
        print("Cannot compare features, one or both are None.")
        return None, None
    if feat1.shape != feat2.shape:
        print(f"Cannot compare features with different shapes: {feat1.shape} vs {feat2.shape}")
        return None, None

    # Euclidean Distance
    euclidean_dist = norm(feat1 - feat2) # Same as np.sqrt(np.sum((feat1-feat2)**2))

    # Cosine Similarity
    cosine_sim = np.dot(feat1, feat2) / (norm(feat1) * norm(feat2))

    return euclidean_dist, cosine_sim

# --- Main Execution ---
def main():
    """
    Main function to load I3D model, extract features for two images, and compare them.
    """
    print(f"Current Working Directory: {os.getcwd()}")
    print(f"Using device: {DEVICE}")

    # --- Load I3D Model ---
    i3d_model = load_i3d_model(I3D_WEIGHTS_PATH)
    if i3d_model is None:
        sys.exit(1) # Exit if model loading failed

    # --- Setup I3D Hook ---
    # This hook will capture the output of the 'avgpool' layer during forward pass
    i3d_features_list = []
    def i3d_hook(module, input, output):
        i3d_features_list.append(output.clone().detach())

    try:
        # Register the forward hook to the average pooling layer
        handle = i3d_model.avgpool.register_forward_hook(i3d_hook)
    except AttributeError:
        print("Error: Could not find 'avgpool' layer in the I3D model.")
        sys.exit(1)

    # --- Extract Features for Image 1 ---
    features1 = extract_features_from_image(IMAGE_PATH_1, i3d_model, i3d_transform, i3d_features_list)

    # --- Extract Features for Image 2 ---
    features2 = extract_features_from_image(IMAGE_PATH_2, i3d_model, i3d_transform, i3d_features_list)

    # --- Cleanup I3D Hook ---
    handle.remove() # Important to remove the hook when done

    # --- Compare Features ---
    print("\n--- Feature Comparison ---")
    dist, sim = compare_features(features1, features2)

    if dist is not None and sim is not None:
        print(f"Features for '{IMAGE_PATH_1}' vs '{IMAGE_PATH_2}':")
        print(f"  - Euclidean Distance: {dist:.4f}")
        print(f"  - Cosine Similarity:  {sim:.4f}")
        print("\nInterpretation:")
        print("  - Euclidean Distance: Lower values mean features are more similar.")
        print("  - Cosine Similarity: Values closer to 1 mean features are more similar (point in the same direction).")
        print("                       Values closer to 0 mean features are less related (orthogonal).")
        print("                       Values closer to -1 mean features are opposite.")
    else:
        print("Comparison could not be performed due to errors during feature extraction.")


if __name__ == "__main__":
    main()
