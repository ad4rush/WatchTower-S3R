# Video Surveillance System - User Guide

This guide walks you through how to set up and use the Video Surveillance System with S3R anomaly detection.

## Getting Started

### System Requirements

- Modern web browser with camera access (Chrome, Firefox, Edge recommended)
- Stable internet connection
- Camera device (built-in or external webcam)

### Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/video-surveillance-system.git
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up the PostgreSQL database:
   ```
   # Make sure PostgreSQL is running
   # Create a database
   # Update DATABASE_URL in the environment variables
   ```

4. Start the application:
   ```
   python main.py
   ```

5. Navigate to `http://localhost:5000` in your browser.

## Using the Application

### Home Page

The home page provides an overview of your surveillance system, including:
- System uptime
- Latest statistics
- Quick links to the surveillance page

### Surveillance Page

The surveillance page is where you monitor and detect anomalies:

1. **Starting Surveillance**:
   - Click the "Allow" button when prompted to give camera access.
   - The video feed will automatically start.

2. **Toggle Surveillance**:
   - Use the "Toggle Surveillance" button to pause or resume the video feed.
   - When paused, the system will not process frames or detect anomalies.
   - The button will change color to indicate the current state.

3. **Anomaly Detection**:
   - The system automatically processes video frames to detect anomalies.
   - When an anomaly is detected, an alarm sound will play.
   - The system will display the anomaly with a backward timer replay.

4. **Alarm Controls**:
   - Click the "Reset Alarm" button to silence an active alarm.

## Understanding Anomaly Detection

### How It Works

The system uses the S3R (Self-Supervised Sparse Representation) model to detect anomalies:

1. Video frames are captured from your camera.
2. Frames are processed and analyzed by the S3R model.
3. The model compares frames against learned normal patterns.
4. Deviations from normal patterns are flagged as anomalies.
5. Alerts are triggered when anomalies are detected.

### Confidence Scores

Each anomaly detection comes with a confidence score:
- **High confidence (0.8-1.0)**: Very likely to be an anomaly
- **Medium confidence (0.6-0.8)**: Possibly an anomaly
- **Low confidence (0.5-0.6)**: Borderline case, may be a false positive

### False Positives

The system may occasionally flag normal activities as anomalies (false positives). This is normal and expected in anomaly detection systems. Common causes include:
- Sudden lighting changes
- Fast camera movements
- New objects appearing in the frame

## Advanced Configuration

### Detection Threshold

The default detection threshold is 0.6. This can be adjusted in the `ml_model.py` file:

```python
self.detection_threshold = 0.6  # Adjust this value
```

- Lower values (e.g., 0.5) will make the system more sensitive, detecting more anomalies but also more false positives.
- Higher values (e.g., 0.8) will make the system more conservative, reducing false positives but potentially missing some anomalies.

### Frame Processing

By default, the system processes every 30th frame to balance performance and detection quality:

```python
self.process_every_n_frames = 30  # Process every 30th frame
```

Adjust this value based on your hardware capabilities:
- Lower values provide more responsive detection but require more processing power.
- Higher values reduce system load but may miss short-duration anomalies.

## Troubleshooting

### Camera Access Issues

If the browser doesn't show a camera feed:
1. Ensure you've granted camera permissions.
2. Check if another application is using the camera.
3. Try refreshing the page.
4. Restart your browser if problems persist.

### Performance Issues

If the system is running slowly:
1. Close other resource-intensive applications.
2. Try using a more powerful device.
3. Consider adjusting the frame processing rate as described above.

### Detection Problems

If anomaly detection isn't working as expected:
1. Ensure your camera has adequate lighting.
2. Try to keep the camera stable.
3. Consider adjusting the detection threshold.

## Privacy Considerations

This system processes video data from your camera. Please be mindful of:
- Only using the system in areas where video surveillance is permitted.
- Informing others if they may be captured by the surveillance system.
- Following local laws and regulations regarding video surveillance.

## Getting Help

If you encounter issues or have questions:
1. Check the troubleshooting section above.
2. Review the code documentation in the repository.
3. Open an issue on GitHub for technical problems.
4. Contact the development team for further assistance.