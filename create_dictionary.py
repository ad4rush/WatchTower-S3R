"""
Create a simplified dictionary file for S3R model
This is a temporary solution until we can download the full pre-trained weights
"""

import numpy as np
import os
from pathlib import Path

# Feature dimension (UCF-Crime features are 2048-dimensional)
FEATURE_DIM = 2048
# Number of dictionary atoms
DICT_SIZE = 100

def create_dictionary():
    # Create directory if it doesn't exist
    dict_dir = Path("dictionary/ucf-crime")
    dict_dir.mkdir(parents=True, exist_ok=True)
    
    # Output file path
    output_path = dict_dir / "ucf-crime_dictionaries.taskaware.omp.100iters.50pct.npy"
    
    # Generate random dictionary (in practice, this would be learned from normal videos)
    dictionary = np.random.randn(DICT_SIZE, FEATURE_DIM).astype(np.float32)
    
    # Normalize each dictionary atom to unit norm (as in the original implementation)
    norms = np.linalg.norm(dictionary, axis=1, keepdims=True)
    dictionary = dictionary / norms
    
    # Save the dictionary
    np.save(output_path, dictionary)
    print(f"Created dictionary with shape {dictionary.shape} at {output_path}")

if __name__ == "__main__":
    create_dictionary()