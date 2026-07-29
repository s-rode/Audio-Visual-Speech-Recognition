"""
Video preprocessing utilities for SwinLip inference
"""
import cv2
import numpy as np
import torch

__all__ = ['Compose', 'Normalize', 'CenterCrop', 'RgbToGray', 'NormalizeUtterance', 'preprocess_video']


class Compose(object):
    """Compose several preprocessing transforms together."""
    
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for transform in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(transform)
        format_string += '\n)'
        return format_string


class RgbToGray(object):
    """Convert RGB frames to grayscale."""

    def __call__(self, frames):
        """
        Args:
            frames (numpy.ndarray): RGB frames to be converted to grayscale
        Returns:
            numpy.ndarray: Grayscale frames
        """
        frames = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames], axis=0)
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize frames with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            frames (numpy.ndarray): Frames to be normalized
        Returns:
            numpy.ndarray: Normalized frames
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop frames at the center."""
    
    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            frames (numpy.ndarray): Frames to be center cropped (T, H, W)
        Returns:
            numpy.ndarray: Center cropped frames
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.)
        delta_h = int(round((h - th)) / 2.)
        frames = frames[:, delta_h:delta_h+th, delta_w:delta_w+tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class NormalizeUtterance(object):
    """Normalize audio utterance by removing mean and dividing by standard deviation."""
    
    def __call__(self, signal):
        """
        Args:
            signal (numpy.ndarray): Audio signal to be normalized
        Returns:
            numpy.ndarray: Normalized audio signal
        """
        signal_std = 0. if np.std(signal) == 0. else np.std(signal)
        signal_mean = np.mean(signal)
        return (signal - signal_mean) / signal_std

    def __repr__(self):
        return self.__class__.__name__ + '()'


def resize_frame(frame, target_size=(88, 88)):
    """Resize single frame to target size."""
    return cv2.resize(frame, target_size)


def temporal_pad(frames, target_length=29):
    """Pad or truncate frames to target temporal length."""
    current_length = len(frames)
    
    if current_length > target_length:
        # Truncate from center
        start_idx = (current_length - target_length) // 2
        return frames[start_idx:start_idx + target_length]
    elif current_length < target_length:
        # Pad with last frame repetition
        padding_length = target_length - current_length
        last_frame = frames[-1:] if len(frames) > 0 else frames[:1]
        padding = np.repeat(last_frame, padding_length, axis=0)
        return np.concatenate([frames, padding], axis=0)
    
    return frames


def preprocess_video(frames, target_size=(88, 88), target_length=29):
    """
    Complete video preprocessing pipeline for SwinLip inference
    
    Args:
        frames: numpy array of shape (T, H, W) or (T, H, W, C)
        target_size: target spatial size (height, width)
        target_length: target temporal length
        
    Returns:
        torch.Tensor: Preprocessed tensor of shape (1, 1, T, H, W)
    """
    # Convert to grayscale if RGB/RGBA
    if len(frames.shape) == 4:
        if frames.shape[-1] == 3:
            frames = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames])
        elif frames.shape[-1] == 4:
            frames = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY) for frame in frames])
    
    # Ensure correct shape (T, H, W)
    if len(frames.shape) != 3:
        raise ValueError(f"Expected 3D frames (T, H, W), got {frames.shape}")
    
    # Resize frames to target spatial size
    resized_frames = []
    for frame in frames:
        resized_frame = resize_frame(frame, target_size)
        resized_frames.append(resized_frame)
    frames = np.stack(resized_frames)
    
    # Temporal padding/truncation
    frames = temporal_pad(frames, target_length)
    
    # Ensure frames are in [0, 255] range
    if frames.max() <= 1.0:
        frames = frames * 255.0
    
    # Convert to float32
    frames = frames.astype(np.float32)
    
    # Convert to tensor and add batch/channel dimensions
    tensor = torch.FloatTensor(frames)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (B, C, T, H, W) = (1, 1, T, H, W)
    
    return tensor


def get_inference_preprocessing():
    """
    Get standard preprocessing pipeline for inference
    
    Returns:
        dict: Dictionary with preprocessing pipelines
    """
    # Standard preprocessing for LRW evaluation (matches original training preprocessing)
    test_pipeline = Compose([
        Normalize(0.0, 255.0),      # Scale to [0,1] 
        CenterCrop((88, 88)),       # Center crop to 88x88
        Normalize(0.421, 0.165),    # Normalize with dataset statistics
    ])
    
    return {
        'test': test_pipeline,
        'inference': preprocess_video
    }