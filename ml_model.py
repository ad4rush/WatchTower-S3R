import cv2
import numpy as np
import os
import torch
import logging
import time
from datetime import datetime
from pathlib import Path

# Import our S3R model implementation
from ml_models.s3r_model import load_s3r_model
from ml_models.s3r_inference_wrapper import S3RInferenceWrapper

logger = logging.getLogger(__name__)

class S3RAnomalyDetector:
    """S3R-based anomaly detector for video surveillance"""
    
    def __init__(self):
        """Initialize the S3R-based anomaly detector"""
        # Configuration
        self.detection_threshold = 0.6  # Confidence threshold for anomaly detection
        self.frame_count = 0
        self.process_every_n_frames = 2 # Only process every Nth frame
        self.cooldown_frames = 30 # Cooldown period after detection (3 seconds at 30fps)
        self.cooldown_counter = 0
        
        # Initialize the device (use CUDA if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        logger.info(f"Using device: {self.device}")
        
        # Set up model paths
        root_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        dictionary_path = root_dir / "dictionary" / "ucf-crime" / "ucf-crime_dictionaries.taskaware.omp.100iters.50pct.npy"
        model_path = root_dir / "checkpoint" / "ucf-crime_s3r_i3d_best.pth"
        
        if not os.path.exists(dictionary_path):
            logger.warning(f"Dictionary not found at {dictionary_path}")
            dictionary_path = None
        
        if not os.path.exists(model_path):
            logger.warning(f"Model weights not found at {model_path}")
            model_path = None
        
        # Load the S3R model
        try:
            self.model = S3RInferenceWrapper(
                checkpoint_path=str(model_path),
                dictionary_path=str(dictionary_path),
                feature_dim=2048,
                quantize_size=32,
                batch_size=1,
                device=self.device,
                modality='task-aware'
            )
            logger.info("S3R model loaded successfully (wrapper)")
        except Exception as e:
            logger.error(f"Error loading S3R model: {e}")
            # Fallback to background subtraction if model fails to load
            self.model = None
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, 
                varThreshold=16, 
                detectShadows=False
            )
            self.motion_threshold = 0.05
            logger.warning("Falling back to background subtraction method")
        
        logger.info("S3R Anomaly detector initialized")
    
    def _detect_with_background_subtraction(self, frame):
        return None, 0.0, "Background subtraction not implemented"
    
    def detect(self, frame):
        """
        Detect anomalies in a frame using the S3R model
        
        Args:
            frame: OpenCV image frame
            
        Returns:
            tuple: (is_anomaly, confidence, description)
        """
        try:
            # Increment frame counter
            self.frame_count += 1
            
            # Apply cooldown if needed
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return False, 0.0, "Cooldown active"
            
            # Skip frames to reduce processing load
            if self.frame_count % self.process_every_n_frames != 0:
                return False, 0.0, "Frame skipped"
            
            # If model failed to load, use background subtraction
            if self.model is None:
                return self._detect_with_background_subtraction(frame)
            
            # --- S3R feature extraction and inference ---
            if not hasattr(self, 'frame_buffer'):
                self.frame_buffer = []
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) < 32:
                return False, 0.0, f"Buffering frames ({len(self.frame_buffer)}/32)"
            
            # Extract I3D features for the 32-frame buffer
            logger.info("Extracting I3D features for S3R inference (buffer size: 32)...")
            model, transform, device = get_i3d_model()
            features = extract_i3d_features_from_frames(self.frame_buffer, model, transform, device)
            logger.info(f"I3D features extracted. Shape: {features.shape}")
            self.frame_buffer = []  # Reset buffer
            
            # Ensure features shape is [32, 2048]
            if features.shape == (2048,):
                features = features.reshape(1, 2048)
            if features.shape[0] != 32:
                # Pad or repeat to 32 if needed
                pad_len = 32 - features.shape[0]
                if pad_len > 0:
                    features = np.pad(features, ((0, pad_len), (0, 0)), mode='edge')
                else:
                    features = features[:32]
            
            logger.info(f"Running S3R inference on features of shape: {features.shape}")
            score = self.model.infer(features)
            logger.info(f"S3R inference complete. Anomaly score: {score:.4f}")
            is_anomaly = score > self.detection_threshold
            description = f"Anomalous activity detected with {score:.2f} confidence" if is_anomaly else "Normal activity"
            
            if is_anomaly:
                self.cooldown_counter = self.cooldown_frames
                logger.info(f"S3R detected anomaly: {description}")
            
            return is_anomaly, score, description
            
        except Exception as e:
            logger.error(f"Error in S3R anomaly detection: {str(e)}")
            # Fallback to background subtraction
            return self._detect_with_background_subtraction(frame)

def extract_i3d_features_from_frames(snippets, model, transform, device):
    """
    Extract I3D features from a list of snippets (each snippet is a list of 16 frames).
    Args:
        snippets: list of 32 lists, each with 16 frames (BGR numpy arrays)
        model: loaded I3D model (eval mode)
        transform: preprocessing function (from extract_i3d_features.py)
        device: 'cuda' or 'cpu'
    Returns:
        features: numpy array of shape [32, 2048]
    """
    import torch
    import cv2
    import numpy as np

    features_list = []
    for snippet in snippets:
        # Preprocess and stack
        clip = torch.stack([transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in snippet])  # [T, 3, 224, 224]
        clip = clip.permute(1, 0, 2, 3).unsqueeze(0).to(device)    # [1, 3, 16, 224, 224]
        # Extract features
        temp_features = []
        def hook(module, input, output):
            temp_features.append(output.clone().detach())
        handle = model.avgpool.register_forward_hook(hook)
        model.eval()
        with torch.no_grad():
            temp_features.clear()
            _ = model({'frames': clip})
            feature = temp_features[0]
            feature = feature.squeeze().cpu().numpy()
        handle.remove()
        features_list.append(feature)
    features = np.stack(features_list)  # [32, 2048]
    return features

def get_i3d_model():
    import torch
    from torchvision import transforms
    import sys
    import os
    # Add pytorch_resnet3d/models to sys.path
    resnet3d_path = os.path.join(os.path.dirname(__file__), 'pytorch_resnet3d', 'models')
    if resnet3d_path not in sys.path:
        sys.path.insert(0, resnet3d_path)
    from resnet import i3_res50
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), 'pretrained', 'i3d_r50_kinetics.pth')
    model = i3_res50(num_classes=400)
    weights = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    
    if "state_dict" in weights:
        model.load_state_dict(weights["state_dict"])
    else:
        model.load_state_dict(weights)
        
    model.to(DEVICE)
    model.eval()
    
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    
    return model, transform, DEVICE


_detector = None

def detect_anomaly(frame):
    """
    Detect anomalies in a video frame
    
    Args:
        frame: OpenCV image frame
        
    Returns:
        tuple: (is_anomaly, confidence, description)
    """
    global _detector
    
    
    if _detector is None:
        _detector = S3RAnomalyDetector()
    
    return _detector.detect(frame)
