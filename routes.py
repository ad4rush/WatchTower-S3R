import os
import base64
import cv2
import numpy as np
import json
from datetime import datetime
import logging
from flask import (
    Blueprint, render_template, request, jsonify, flash,
    redirect, url_for, current_app
)

# Assuming models.py and utils.py are correctly structured and importable
# If db operations are needed, db instance should be initialized in app.py and accessed via current_app
# from .models import Anomaly, db # Example if using SQLAlchemy with app factory
# from .utils import save_anomaly_clip, get_anomaly_clips, format_timestamp

logger = logging.getLogger(__name__)

# Create the Blueprint
main_bp = Blueprint('main', __name__, template_folder='templates', static_folder='static')

# --- Routes ---

# Example context processor if needed within the blueprint
# @main_bp.context_processor
# def inject_now():
#     return {'now': datetime.now()}

@main_bp.route('/')
def home():
    # Example data fetching - adapt if using a database
    anomaly_count = 0 # Replace with actual count if db is configured
    last_activity = "No activity yet" # Replace if db is configured
    # try:
    #     # Access db via current_app if initialized in create_app
    #     db = current_app.config.get('DB_INSTANCE') # Example key
    #     if db:
    #         anomaly_count = db.session.query(Anomaly).count()
    #         last_anomaly = db.session.query(Anomaly).order_by(Anomaly.timestamp.desc()).first()
    #         if last_anomaly:
    #             last_activity = format_timestamp(last_anomaly.timestamp.timestamp())
    # except Exception as e:
    #     logger.error(f"Error fetching data for home page: {e}")

    return render_template('home.html',
                           anomaly_count=anomaly_count,
                           last_activity=last_activity)

@main_bp.route('/surveillance')
def surveillance():
    # This route likely needs associated JS to send frames to /process_frame
    return render_template('surveillance.html')

@main_bp.route('/process_frame', methods=['POST'])
def process_frame():
    # Get models and processor from app context (set in create_app)
    s3r_wrapper = current_app.config.get('S3R_WRAPPER')
    video_processor = current_app.config.get('VIDEO_PROCESSOR')
    # db = current_app.config.get('DB_INSTANCE') # Get db instance if needed

    if not s3r_wrapper or not video_processor:
        logger.error("S3R model or VideoProcessor not found in app context.")
        return jsonify({"error": "Server configuration error"}), 500

    try:
        data = request.json
        if not data or 'frame' not in data:
            logger.error("No frame data in request")
            return jsonify({'error': 'Invalid request data'}), 400

        frame_data = data['frame']
        # logger.debug(f"Received frame_base64 (first 100 chars): {frame_data[:100] if frame_data else 'None'}")
        if not frame_data or ',' not in frame_data:
            logger.error("Received empty or malformed frame_base64")
            return jsonify({'error': 'Empty or malformed frame data'}), 400

        timestamp = data.get('timestamp', datetime.now().timestamp())

        # --- Decode Frame ---
        try:
            encoded_data = frame_data.split(',')[1]
            decoded = base64.b64decode(encoded_data)
            nparr = np.frombuffer(decoded, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("cv2.imdecode returned None")
        except Exception as e:
            logger.error(f"Error decoding frame: {e}", exc_info=True)
            return jsonify({'error': f'Failed to decode frame: {e}'}), 400

        # --- Add Frame to Buffer ---
        video_processor.add_frame(frame)
        # logger.debug(f"Buffered frame. Current buffer size: {video_processor.get_buffer_size()}")

        # --- Attempt Inference ---
        is_anomaly = False
        confidence = 0.0
        description = "Buffering..."
        anomaly_id = None
        saved_path = None

        # Call the new method to get the correctly shaped feature sequence
        features = video_processor.extract_features_for_s3r() # Returns [T, C] array or None

        if features is not None:
            logger.info(f"Attempting S3R inference with features shape: {features.shape}")
            try:
                # Perform S3R Inference using the wrapper from app context
                score = s3r_wrapper.infer(features) # Pass [T, C] features
                confidence = float(score) # Ensure it's a standard float
                # Define your anomaly threshold
                anomaly_threshold = 0.5 # Example threshold, adjust as needed
                is_anomaly = confidence > anomaly_threshold
                description = f"Anomaly Score: {confidence:.4f}"
                logger.info(f"S3R inference complete. Score: {confidence:.4f}, Anomaly: {is_anomaly}")

                # --- Save Anomaly (if detected and db is configured) ---
                # if is_anomaly and db:
                #     try:
                #         # Use the most recent frame for the snapshot
                #         saved_path = save_anomaly_clip(frame, timestamp) # Assuming save_anomaly_clip exists in utils
                #         new_anomaly = Anomaly(
                #             timestamp=datetime.fromtimestamp(timestamp),
                #             confidence=confidence,
                #             path=saved_path,
                #             description=f"Anomaly detected (Score: {confidence:.2f})"
                #         )
                #         db.session.add(new_anomaly)
                #         db.session.commit()
                #         anomaly_id = new_anomaly.id
                #         logger.info(f"Anomaly saved to DB (ID: {anomaly_id}) and file ({saved_path})")
                #     except Exception as db_err:
                #         logger.error(f"Error saving anomaly: {db_err}", exc_info=True)
                #         db.session.rollback() # Rollback on error
                # elif is_anomaly:
                #      logger.warning("Anomaly detected but database is not configured for saving.")


            except Exception as e:
                logger.error(f"Error during S3R inference call: {e}", exc_info=True)
                description = "Inference error"
                confidence = -1.0 # Indicate error

        # --- Prepare Response ---
        result = {
            'anomaly': is_anomaly,
            'confidence': confidence,
            'description': description, # Send back the score or status
            'timestamp': timestamp,
            # 'anomaly_id': anomaly_id, # Include if saved
            # 'path': saved_path # Include if saved
        }
        return jsonify(result)

    except Exception as e:
        logger.error(f"Unhandled error in /process_frame: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500


@main_bp.route('/get_anomalies')
def get_anomalies():
    """Get a list of recent anomalies for frontend display"""
    # db = current_app.config.get('DB_INSTANCE') # Get db instance if needed
    anomalies = []
    # if db:
    #     try:
    #         anomalies_db = db.session.query(Anomaly).order_by(Anomaly.timestamp.desc()).limit(20).all()
    #         anomalies = [anomaly.to_dict() for anomaly in anomalies_db] # Assuming to_dict method exists
    #     except Exception as e:
    #         logger.error(f"Error retrieving anomalies from DB: {e}")
    # else:
    #     logger.warning("Database not configured, cannot retrieve anomalies.")

    # Optionally add logic here to read from filesystem if needed, like in your previous version
    # file_anomalies = get_anomaly_clips() # Assuming function exists
    # ... combine logic ...

    return jsonify({'anomalies': anomalies})

# Add other routes from your previous version if needed (e.g., /video_feed)
# Ensure they also use current_app to access shared resources if necessary.

