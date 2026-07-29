#!/usr/bin/env python3
"""
SwinLip Evaluation on LRW Test Set
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings(
    "ignore",
    message="torch.meshgrid"
)
import time
import numpy as np
from tqdm import tqdm

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.model_utils import load_model, load_word_labels
from utils.dataloaders import get_data_loaders, get_preprocessing_pipelines
from utils.dataset import MyDataset, pad_packed_collate


def original_evaluate(model, dset_loader, criterion, use_boundary=False):
    """
    Evaluate model using the EXACT same method as main.py
    """
    model.eval()

    running_loss = 0.
    running_corrects = 0.

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dset_loader)):
            if use_boundary:
                input, lengths, labels, boundaries = data
                boundaries = boundaries.cuda()
            else:
                input, lengths, labels = data
                boundaries = None

            # EXACT same forward pass as original main.py line 144
            logits = model(input.unsqueeze(1).cuda(), lengths=lengths, boundaries=boundaries)
            
            # EXACT same prediction as original main.py line 145
            _, preds = torch.max(F.softmax(logits, dim=1).data, dim=1)
            running_corrects += preds.eq(labels.cuda().view_as(preds)).sum().item()

            loss = criterion(logits, labels.cuda())
            running_loss += loss.item() * input.size(0)

    print(f"{len(dset_loader.dataset)} in total\tCR: {running_corrects/len(dset_loader.dataset)}")
    return running_corrects/len(dset_loader.dataset), running_loss/len(dset_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description='Original-style SwinLip Evaluation')
    parser.add_argument('--config', required=True, help='Path to model config JSON')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', default='LRW/LRW_cropped_gray/', 
                       help='Path to LRW test data')
    parser.add_argument('--label-path', default='labels/500WordsSortedList.txt',
                       help='Path to word labels file')
    parser.add_argument('--annonation-direc', default='LRW/lipread_mp4',
                       help='Path to annotation directory')
    parser.add_argument('--batch-size', type=int, default=32, 
                       help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of data loader workers')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    print("="*60)
    print("Original-style SwinLip LRW Evaluation")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Batch Size: {args.batch_size}")
    print()
    
    # Load model
    print("Loading model...")
    start_time = time.time()
    model, config = load_model(args.config, args.checkpoint, 'cuda')
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f}s")
    print()
    
    # Create args object exactly like original main.py
    class Args:
        def __init__(self):
            self.modality = 'video'
            self.data_dir = args.data_dir
            self.label_path = args.label_path
            self.annonation_direc = args.annonation_direc
            self.use_boundary = config.get('use_boundary', False)
            self.test = True
            self.batch_size = args.batch_size
            self.workers = args.workers
    
    eval_args = Args()
    
    print("Loading test data with original preprocessing...")
    
    # Use EXACT same preprocessing pipeline as original
    preprocessing = get_preprocessing_pipelines(eval_args.modality)
    
    # Create dataset using EXACT same parameters as original
    test_dataset = MyDataset(
        modality=eval_args.modality,
        data_partition='test',
        data_dir=eval_args.data_dir,
        label_fp=eval_args.label_path,
        annonation_direc=eval_args.annonation_direc,
        preprocessing_func=preprocessing['test'],
        data_suffix='.npz',
        use_boundary=eval_args.use_boundary,
    )
    
    # Create data loader using EXACT same parameters as original
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_args.batch_size,
        shuffle=False,  # Don't shuffle for evaluation
        collate_fn=pad_packed_collate,
        pin_memory=True,
        num_workers=eval_args.workers,
        worker_init_fn=np.random.seed(1)
    )
    
    print(f"Loaded {len(test_dataset)} test samples")
    print()
    
    # Create criterion (same as original)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Run evaluation using EXACT same method as original main.py
    print("Starting evaluation...")
    start_time = time.time()
    accuracy, avg_loss = original_evaluate(
        model, test_loader, criterion, eval_args.use_boundary
    )
    eval_time = time.time() - start_time
    
    # Print results
    model_name = config.get('model_name', 'SwinLip')
    if config.get('use_boundary', False):
        model_name += " (Word Boundary)"
    
    print("\n" + "="*60)
    print(f"{model_name} Evaluation Results (Original Method)")
    print("="*60)
    print(f"Accuracy: {accuracy:.6f} ({accuracy*100:.2f}%)")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Total Samples: {len(test_dataset):,}")
    print(f"Correct Predictions: {int(accuracy * len(test_dataset)):,}")
    print(f"Evaluation Time: {eval_time:.2f}s")
    print(f"Speed: {len(test_dataset)/eval_time:.1f} samples/s")
    
    print("\n* This uses the EXACT same evaluation method as main.py --test")


if __name__ == "__main__":
    main()