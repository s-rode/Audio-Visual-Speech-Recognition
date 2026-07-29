"""
Model loading utilities for SwinLip inference
"""
import torch
import json
from models.swinlip import SwinLip


def load_config(config_path):
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_model(config_path, checkpoint_path, device='cuda'):
    """
    Load SwinLip model with pretrained weights
    
    Args:
        config_path: Path to model configuration JSON
        checkpoint_path: Path to model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model in evaluation mode
    """
    # Load config
    config = load_config(config_path)
    
    # Create model
    model = SwinLip(config)
    
    # Load checkpoint
    if device == 'cuda' and torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
        model = model.cuda()
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        device = 'cpu'
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model: {config.get('model_name', 'SwinLip')}")
    print(f"Use boundary: {config.get('use_boundary', False)}")
    
    return model, config


def load_word_labels(labels_path):
    """Load word labels from text file"""
    with open(labels_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels