import torch
import numpy as np
from .preprocess_inference import *
from .dataset import MyDataset, pad_packed_collate


def get_preprocessing_pipelines(modality):
    # -- preprocess for the video stream (inference/evaluation only)
    preprocessing = {}
    # -- LRW config
    if modality == 'video':
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)
        
        # Only keep evaluation/test preprocessing (no training augmentations)
        test_preprocessing = Compose([
            Normalize(0.0, 255.0),    # Scale to [0,1]
            CenterCrop(crop_size),    # Center crop to 88x88
            Normalize(mean, std)      # Dataset normalization
        ])
        
        preprocessing['val'] = test_preprocessing
        preprocessing['test'] = test_preprocessing

    elif modality == 'audio':
        # Audio preprocessing for inference/evaluation only
        audio_preprocessing = NormalizeUtterance()  # Only normalization, no noise augmentation
        
        preprocessing['val'] = audio_preprocessing
        preprocessing['test'] = audio_preprocessing

    return preprocessing


def get_data_loaders(args):
    preprocessing = get_preprocessing_pipelines( args.modality)

    # create dataset object for each partition
    partitions = ['test'] if args.test else ['train', 'val', 'test']
    dsets = {partition: MyDataset(
                modality=args.modality,
                data_partition=partition,
                data_dir=args.data_dir,
                label_fp=args.label_path,
                annonation_direc=args.annonation_direc,
                preprocessing_func=preprocessing[partition],
                data_suffix='.npz',
                use_boundary=args.use_boundary,
                ) for partition in partitions}
    dset_loaders = {x: torch.utils.data.DataLoader(
                        dsets[x],
                        batch_size=args.batch_size,
                        shuffle=True,
                        collate_fn=pad_packed_collate,
                        pin_memory=True,
                        num_workers=args.workers,
                        worker_init_fn=np.random.seed(1)) for x in partitions}
    return dset_loaders
