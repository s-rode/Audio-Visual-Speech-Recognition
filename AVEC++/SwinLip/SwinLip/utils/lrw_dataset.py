"""
LRW Dataset loader for evaluation
"""
import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .preprocess_inference import preprocess_video


class LRWTestDataset(Dataset):
    """LRW Test Dataset for evaluation"""
    
    def __init__(self, data_dir, labels_path, use_boundary=False, annotation_dir=None):
        self.data_dir = data_dir
        self.use_boundary = use_boundary
        self.annotation_dir = annotation_dir
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        # Create label to index mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        # Load all test files
        self.data_files = []
        self._load_test_files()
        
        print(f"Loaded {len(self.data_files)} test samples from {len(self.labels)} classes")
    
    def _load_test_files(self):
        """Load all test files from LRW dataset"""
        for label in self.labels:
            label_dir = os.path.join(self.data_dir, label, 'test')
            if os.path.exists(label_dir):
                files = glob.glob(os.path.join(label_dir, '*.npz'))
                for file_path in files:
                    self.data_files.append({
                        'path': file_path,
                        'label': label,
                        'label_idx': self.label_to_idx[label]
                    })
    
    def _get_boundary(self, file_path):
        """Get boundary information for word boundary model"""
        if not self.use_boundary or not self.annotation_dir:
            return None
            
        # Convert data path to annotation path
        rel_path = os.path.relpath(file_path, self.data_dir)
        annotation_path = os.path.join(self.annotation_dir, rel_path)
        annotation_path = os.path.splitext(annotation_path)[0] + '.txt'
        
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            
            # Parse utterance duration from annotation
            utterance_duration = float(lines[4].split(' ')[1])
            
            # Load actual video to get frame count
            data = np.load(file_path)['data']
            n_frames = len(data)
            
            # Calculate boundary 
            fps = 25
            half_interval = int(utterance_duration / 2.0 * fps)
            mid_idx = (n_frames - 1) // 2
            left_idx = max(0, mid_idx - half_interval - 1)
            right_idx = min(mid_idx + half_interval + 1, n_frames)
            
            boundary = np.zeros(n_frames)
            boundary[left_idx:right_idx] = 1
            
            return boundary
            
        except Exception as e:
            print(f"Warning: Could not load boundary for {file_path}: {e}")
            # Return dummy boundary (all speech)
            data = np.load(file_path)['data'] 
            return np.ones(len(data))
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        file_info = self.data_files[idx]
        
        # Load video data
        data = np.load(file_info['path'])['data']
        
        # Preprocess video (but keep original preprocessing for LRW)
        # LRW data is already preprocessed, so we mainly need tensor conversion
        if len(data.shape) == 3:  # T, H, W
            # Ensure data is in correct format
            if data.shape[1] != 88 or data.shape[2] != 88:
                # Resize if needed
                from .preprocess_inference import resize_frame
                resized_data = []
                for frame in data:
                    resized_frame = resize_frame(frame, (88, 88))
                    resized_data.append(resized_frame)
                data = np.array(resized_data)
            
            # Temporal padding to 29 frames if needed
            from .preprocess_inference import temporal_pad
            data = temporal_pad(data, 29)
            
            # Convert to tensor
            video_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(0)  # (1, 1, T, H, W)
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        # Get boundary if needed
        boundary = None
        if self.use_boundary:
            boundary_data = self._get_boundary(file_info['path'])
            if boundary_data is not None:
                # Match the length with processed video
                from .preprocess_inference import temporal_pad
                boundary_padded = temporal_pad(boundary_data.reshape(-1, 1, 1), 29).squeeze()
                boundary = torch.FloatTensor(boundary_padded).unsqueeze(-1)  # (T, 1)
        
        return {
            'video': video_tensor.squeeze(0),  # (1, T, H, W)
            'label': file_info['label_idx'],
            'boundary': boundary,
            'path': file_info['path'],
            'word': file_info['label']
        }


def collate_fn(batch):
    """Custom collate function for LRW evaluation"""
    videos = []
    labels = []
    boundaries = []
    paths = []
    words = []
    
    for item in batch:
        videos.append(item['video'])
        labels.append(item['label'])
        boundaries.append(item['boundary'])
        paths.append(item['path'])
        words.append(item['word'])
    
    # Stack videos
    videos = torch.stack(videos)  # (B, 1, T, H, W)
    labels = torch.LongTensor(labels)
    
    # Handle boundaries
    if boundaries[0] is not None:
        boundaries = torch.stack(boundaries)  # (B, T, 1)
    else:
        boundaries = None
    
    return {
        'videos': videos,
        'labels': labels,
        'boundaries': boundaries,
        'paths': paths,
        'words': words,
        'lengths': [videos.shape[2]] * len(videos)  # All videos same length after padding
    }


def get_lrw_test_loader(data_dir, labels_path, batch_size=32, use_boundary=False, 
                       annotation_dir=None, num_workers=4):
    """Get LRW test data loader"""
    dataset = LRWTestDataset(
        data_dir=data_dir,
        labels_path=labels_path,
        use_boundary=use_boundary,
        annotation_dir=annotation_dir
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return loader