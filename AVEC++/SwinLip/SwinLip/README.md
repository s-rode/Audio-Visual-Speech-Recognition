# SwinLip for Efficient Lip Reading

This is the official implementation of **SwinLip**, a Swin Transformer-based lipreading model that achieves state-of-the-art performance on the LRW (Lip Reading in the Wild) dataset.

**Paper**: [SwinLip: An Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer](https://arxiv.org/abs/2505.04394)  
**Authors**: Young-Hu Park, Rae-Hong Park, Hyung-Min Park  
**Published**: Neurocomputing, Vol. 639, 2025  

## Setup

### Environment Setup

Create and activate the conda environment:

```bash
conda create -n swinlip python=3.8
conda activate swinlip
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Requirements
- Python >= 3.8
- PyTorch >= 1.8.0
- CUDA-compatible GPU (recommended)

## Dataset Preparation

### LRW Dataset

1. **Download LRW Dataset**: Download the LRW dataset from [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

2. **Preprocessing**: Follow the preprocessing steps from [Lipreading using TCN](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#how-to-prepare-dataset):

```bash
# Install requirements for preprocessing
pip install dlib opencv-python

# Download and run the preprocessing script
# This will extract lip regions and convert to proper format
python preprocessing/extract_lip_from_video.py --dataset LRW --src_dir /path/to/LRW --tgt_dir LRW/
```

3. **Directory Structure**: After preprocessing, organize your data as follows:

```
swinlip-inference/
├── LRW/
│   ├── LRW_cropped_gray/      # Cropped grayscale lip regions (.npz files)
│   │   ├── ABOUT/
│   │   │   ├── test/
│   │   │   │   ├── ABOUT_00001.npz
│   │   │   │   └── ...
│   │   ├── ABSOLUTELY/
│   │   └── ...
│   └── lipread_mp4/           # Annotation files (.txt files)
│       ├── ABOUT/
│       │   ├── test/
│       │   │   ├── ABOUT_00001.txt
│       │   │   └── ...
│       ├── ABSOLUTELY/
│       └── ...
├── labels/
│   └── 500WordsSortedList.txt # Word vocabulary (provided)
├── checkpoints/               # Model weights
├── configs/                   # Model configurations
└── ...
```

*Note: The preprocessing creates two separate folders: `LRW_cropped_gray/` contains `.npz` files with cropped lip regions, and `lipread_mp4/` contains `.txt` files with temporal information.*

## Pretrained Models

We provide two pretrained models trained on the LRW dataset:

| Model | Acc. | Checkpoint | Params (M) | FLOPs (G) |
|-------|----------|------------|--------|------|
| **SwinLip** | 90.66% | [`lrw_swinlip.pyh`](https://drive.google.com/file/d/1t5IIUjEq3GYlgVmeGXyiHuqbBCCJ2ecX/view?usp=drive_link) | 53.84 | 3.40 |
| **SwinLip-WB** | 92.41% | [`lrw_swinlip_wb.pyh`](https://drive.google.com/file/d/1-Jn5cX-oObzNQfYgA5pwFkmFaIsAd9aF/view?usp=drive_link) | - | - |

*WB: Word Boundary model with enhanced accuracy using boundary information*

## Testing

### Evaluate on LRW Test Set

```bash
# SwinLip model
python evaluate.py \
    --config configs/swinlip.json \
    --checkpoint checkpoints/lrw_swinlip.pth \
    --data-dir LRW/LRW_cropped_gray/


# SwinLip with word boundary
python evaluate.py \
    --config configs/swinlip_wb.json \
    --checkpoint checkpoints/lrw_swinlip_wb.pth \
    --data-dir LRW/LRW_cropped_gray/
```

## Inference

### Single Video Inference

```bash
# SwinLip model
python inference.py \
    --config configs/swinlip.json \
    --checkpoint checkpoints/lrw_swinlip.pth \
    --input video.mp4

# SwinLip with word boundary
python inference.py \
    --config configs/swinlip_wb.json \
    --checkpoint checkpoints/lrw_swinlip_wb.pth \
    --input video.mp4
```

### Supported Input Formats

- **Video files**: .mp4, .avi, .mov
- **Numpy arrays**: .npy (shape: T, H, W)
- **Compressed arrays**: .npz with 'data' key



## Citation

If you use SwinLip in your research, please cite our paper:

```bibtex
@article{park2025swinlip,
  title={SwinLip: An Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer},
  author={Park, Young-Hu and Park, Rae-Hong and Park, Hyung-Min},
  journal={Neurocomputing},
  volume={639},
  pages={130289},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgement

This repository is built using the [Lipreading using TCN](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks), [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [KoSpeech](https://github.com/sooftware/kospeech) repositories.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contact

For questions and issues, please open an issue on GitHub.
