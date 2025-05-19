# Video Surveillance System with S3R Anomaly Detection

A robust video surveillance system that leverages advanced machine learning techniques to detect anomalies in real-time video feeds.

## Overview

This system uses the S3R (Self-Supervised Sparse Representation) model for anomaly detection in surveillance videos. It provides a web interface for monitoring video feeds, detecting unusual activities, and alerting users when anomalies are detected.

## Features

- **Real-time anomaly detection** using the S3R model trained on the UCF-Crime dataset
- **Live video monitoring** with browser-based camera access
- **Toggle surveillance** functionality to pause/resume monitoring
- **Visual and audio alerts** when anomalies are detected
- **Backward timer replay** to review what triggered an alert
- **Database storage** of detected anomalies for future reference
- **Web-based interface** for easy access from any device

## Technology Stack

- **Backend**: Python with Flask
- **Machine Learning**: PyTorch, OpenCV
- **Database**: PostgreSQL
- **Frontend**: JavaScript, HTML5, Bootstrap
- **Audio**: Tone.js for alert sounds

## Project Structure

```
.
├── ml_models/            # Machine learning model implementations
│   ├── __init__.py
│   └── s3r_model.py      # S3R model implementation
├── static/               # Static files (JS, CSS, etc.)
│   ├── css/
│   ├── js/
│   └── sounds/
├── templates/            # HTML templates
├── dictionary/           # Dictionary files for S3R model
├── checkpoint/           # Model checkpoint files
├── app.py                # Flask application setup
├── main.py               # Entry point
├── ml_model.py           # ML model wrapper
├── models.py             # Database models
├── routes.py             # Application routes
├── utils.py              # Utility functions
├── video_processor.py    # Video processing utilities
└── create_dictionary.py  # Script to create dictionary files
```

## Theoretical Background

The S3R model is based on the paper "Exploring Sparse Self-Supervised Representation for Video Anomaly Detection" (ECCV 2022). It uses a sparse dictionary learning approach to identify normal patterns in videos and detect deviations from these patterns as anomalies.

Key components:
- **Dictionary Learning**: Creates a sparse representation of normal activities
- **Self-Attention**: Captures temporal relationships in video frames
- **Encoder-Decoder Architecture**: Separates normal from anomalous patterns

## Datasets

The system is designed to work with the UCF-Crime dataset checkpoint, which contains videos of various anomalous activities such as fighting, robbery, and assault.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The S3R model is based on research by [Louis Liu et al.](https://github.com/louisYen/S3R)
- UCF-Crime dataset by the University of Central Florida