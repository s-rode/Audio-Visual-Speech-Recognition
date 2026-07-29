#!/usr/bin/env python3
"""
SwinLip Inference Script
"""

import os
import sys
import argparse
import torch
import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid"
)
import numpy as np
import cv2
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_utils import load_model, load_word_labels
from utils.preprocess_inference import preprocess_video


def load_video(video_path):
    """
    Load video from file
    
    Args:
        video_path: Path to video file (.mp4, .avi, .npy, .npz)
        
    Returns:
        frames: numpy array of shape (T, H, W)
    """
    ext = Path(video_path).suffix.lower()
    
    if ext in ['.mp4', '.avi', '.mov']:
        # Load video file
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        
        cap.release()
        return np.array(frames)
    
    elif ext == '.npy':
        return np.load(video_path)
    
    elif ext == '.npz':
        data = np.load(video_path)
        return data['data']
    
    else:
        raise ValueError(f"Unsupported video format: {ext}")


def predict_single_video(model, config, video_tensor, word_labels, boundaries=None):
    """
    Predict word from single video
    
    Args:
        model: Loaded SwinLip model
        config: Model configuration
        video_tensor: Preprocessed video tensor (1, 1, T, H, W)
        word_labels: List of word labels
        boundaries: Optional boundary tensor for word boundary model
        
    Returns:
        predicted_word: Predicted word string
        confidence: Prediction confidence
        probabilities: All class probabilities
    """
    with torch.no_grad():
        # Get video length
        T = video_tensor.shape[2]
        lengths = [T]
        
        # Move to device
        if torch.cuda.is_available():
            video_tensor = video_tensor.cuda()
            if boundaries is not None:
                boundaries = boundaries.cuda()
        
        # Forward pass
        logits = model(video_tensor, lengths, boundaries)
        
        # Get probabilities
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_idx = torch.max(probabilities, dim=1)
        
        # Convert to CPU and numpy
        predicted_idx = predicted_idx.cpu().item()
        confidence = confidence.cpu().item()
        probabilities = probabilities.cpu().numpy()[0]
        
        predicted_word = word_labels[predicted_idx]
        
        return predicted_word, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(description='SwinLip Inference')
    parser.add_argument('--config', required=True, help='Path to model config JSON')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--labels', default='labels/500WordsSortedList.txt', 
                       help='Path to word labels file')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                       help='Device to run inference on')
    parser.add_argument('--top-k', type=int, default=5, 
                       help='Show top-k predictions')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 50)
    print("SwinLip Lipreading Inference")
    print("=" * 50)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")
    print(f"Device: {args.device}")
    print()
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model, config = load_model(args.config, args.checkpoint, args.device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print()
    
    # Load word labels
    if os.path.exists(args.labels):
        word_labels = load_word_labels(args.labels)
        print(f"Loaded {len(word_labels)} word labels")
    else:
        print("Warning: Labels file not found, using generic labels")
        word_labels = [f"word_{i}" for i in range(config['num_classes'])]
    print()
    
    # Load and preprocess video
    print("Loading video...")
    frames = load_video(args.input)
    print(f"Video shape: {frames.shape}")
    
    print("Preprocessing...")
    video_tensor = preprocess_video(frames)
    print(f"Preprocessed tensor shape: {video_tensor.shape}")
    print()
    
    # Prepare boundaries if needed
    boundaries = None
    if config.get('use_boundary', False):
        print("Model uses word boundaries, but boundary info not provided.")
        print("Using dummy boundaries (all speech).")
        T = video_tensor.shape[2]
        boundaries = torch.ones(1, T, 1)  # All frames are speech
    
    # Run inference
    print("Running inference...")
    start_time = time.time()
    predicted_word, confidence, probabilities = predict_single_video(
        model, config, video_tensor, word_labels, boundaries
    )
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.3f}s")
    print()
    
    # Display results
    print("=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Predicted word: {predicted_word}")
    print(f"Confidence: {confidence:.4f}")
    print()
    
    # Show top-k predictions
    top_k_indices = np.argsort(probabilities)[::-1][:args.top_k]
    print(f"Top-{args.top_k} predictions:")
    for i, idx in enumerate(top_k_indices):
        word = word_labels[idx]
        prob = probabilities[idx]
        print(f"{i+1:2d}. {word:15s} ({prob:.4f})")


if __name__ == "__main__":
    main()