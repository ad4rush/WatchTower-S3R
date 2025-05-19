import os
import cv2
import time
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the application"""
    directories = [
        os.path.join(os.getcwd(), 'static'),
        os.path.join(os.getcwd(), 'static', 'anomalies'),
        os.path.join(os.getcwd(), 'static', 'js'),
        os.path.join(os.getcwd(), 'static', 'css'),
        os.path.join(os.getcwd(), 'static', 'sounds'),
        os.path.join(os.getcwd(), 'static', 'svg'),
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def save_anomaly_clip(frame, timestamp):
    """Save an anomaly clip"""
    anomalies_dir = os.path.join(os.getcwd(), 'static', 'anomalies')
    
    if not os.path.exists(anomalies_dir):
        os.makedirs(anomalies_dir)
    
    # Create a filename with timestamp and unique ID
    filename = f"anomaly_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(anomalies_dir, filename)
    
    try:
        # Save the frame as an image
        cv2.imwrite(filepath, frame)
        logger.info(f"Saved anomaly image to {filepath}")
        
        # Return the relative path
        return os.path.join('static', 'anomalies', filename)
    except Exception as e:
        logger.error(f"Error saving anomaly clip: {str(e)}")
        return None

def format_timestamp(timestamp):
    """Format a timestamp for display"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def get_anomaly_clips():
    """Get a list of all saved anomaly clips"""
    anomalies_dir = os.path.join(os.getcwd(), 'static', 'anomalies') # during training only ? or during inference too 
    clips = []
    
    if os.path.exists(anomalies_dir):
        for filename in os.listdir(anomalies_dir):
            if filename.startswith('anomaly_') and (filename.endswith('.mp4') or filename.endswith('.jpg')):
                filepath = os.path.join('static', 'anomalies', filename)
                # Extract timestamp from filename
                try:
                    timestamp_str = filename.split('_')[1]
                    timestamp = float(timestamp_str)
                    formatted_time = format_timestamp(timestamp)
                    
                    clips.append({
                        'path': filepath,
                        'timestamp': timestamp,
                        'formatted_time': formatted_time
                    })
                except (IndexError, ValueError) as e:
                    logger.error(f"Error parsing filename {filename}: {str(e)}")
    
    # Sort by timestamp (newest first)
    clips.sort(key=lambda x: x['timestamp'], reverse=True)
    return clips
