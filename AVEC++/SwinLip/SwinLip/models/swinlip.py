# Copyright (c) 2025, Young-Hu Park
# All rights reserved.
#
# This source code is licensed under the Apache 2.0 license found in the
# LICENSE file in the root directory of this source tree.
#
# This code is part of the paper:
# "SwinLip: An Efficient Visual Speech Encoder for Lip Reading Using Swin Transformer"
# Young-Hu Park, Rae-Hong Park, Hyung-Min Park
# Neurocomputing, Vol. 639, 2025
# https://arxiv.org/abs/2505.04394


import torch
import torch.nn as nn
import math
from .swinlip_blocks import SwinTransformer
from .densetcn import DenseTemporalConvNet


def threeD_to_2D_tensor(x):
    """Convert 3D tensor to 2D for processing"""
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


def _average_batch(x, lengths, B):
    """Average batch sequences based on lengths"""
    return torch.stack([torch.mean(x[index][:,0:i], 1) for index, i in enumerate(lengths)], 0)


class DenseTCN(nn.Module):
    """Dense Temporal Convolutional Network for sequence modeling"""
    
    def __init__(self, block_config, growth_rate_set, input_size, reduced_size, num_classes,
                 kernel_size_set, dilation_size_set, dropout, squeeze_excitation=False):
        super(DenseTCN, self).__init__()
        
        num_features = reduced_size + block_config[-1] * growth_rate_set[-1]
        self.tcn_trunk = DenseTemporalConvNet(
            block_config, growth_rate_set, input_size, reduced_size,
            kernel_size_set, dilation_size_set,
            dropout=dropout, relu_type='prelu',
            squeeze_excitation=squeeze_excitation,
        )
        self.tcn_output = nn.Linear(num_features, num_classes)
        self.consensus_func = _average_batch

    def forward(self, x, lengths, B):
        x = self.tcn_trunk(x.transpose(1, 2))
        x = self.consensus_func(x, lengths, B)
        return self.tcn_output(x)


class SwinLip(nn.Module):
    """
    SwinLip: Swin Transformer based Lipreading Model
    Inference-only implementation
    """
    
    def __init__(self, config):
        super(SwinLip, self).__init__()
        
        self.modality = 'video'
        self.use_boundary = config.get('use_boundary', False)
        self.num_classes = config['num_classes']
        
        # Frontend 3D Conv
        self.frontend_nout = 24
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, self.frontend_nout, kernel_size=(3, 5, 5), 
                     stride=(1, 1, 1), padding=(1, 2, 2), bias=False),
            nn.BatchNorm3d(self.frontend_nout),
            nn.PReLU(self.frontend_nout),
        )
        
        # Swin Transformer Backbone (only supported backbone for inference)
        self.trunk = SwinTransformer(
            img_size=88,
            patch_size=11,
            in_chans=self.frontend_nout,
            num_classes=512,
            window_size=4
        )
        self.backend_out = 512
        
        # DenseTCN for temporal modeling
        densetcn_options = {
            'block_config': config['densetcn_block_config'],
            'growth_rate_set': config['densetcn_growth_rate_set'],
            'reduced_size': config['densetcn_reduced_size'],
            'kernel_size_set': config['densetcn_kernel_size_set'],
            'dilation_size_set': config['densetcn_dilation_size_set'],
            'squeeze_excitation': config['densetcn_se'],
            'dropout': config['densetcn_dropout'],
        }
        
        input_size = self.backend_out
        if self.use_boundary:
            input_size += 1
            
        self.tcn = DenseTCN(
            block_config=densetcn_options['block_config'],
            growth_rate_set=densetcn_options['growth_rate_set'],
            input_size=input_size,
            reduced_size=densetcn_options['reduced_size'],
            num_classes=self.num_classes,
            kernel_size_set=densetcn_options['kernel_size_set'],
            dilation_size_set=densetcn_options['dilation_size_set'],
            dropout=densetcn_options['dropout'],
            squeeze_excitation=densetcn_options['squeeze_excitation'],
        )
        
        # Initialize weights
        self._initialize_weights_randomly()

    def forward(self, x, lengths, boundaries=None):
        """
        Forward pass for inference
        
        Args:
            x: Input video tensor (B, C, T, H, W)
            lengths: Sequence lengths
            boundaries: Optional word boundary information
        """
        B, C, T, H, W = x.size()
        
        # 3D CNN frontend
        x = self.frontend3D(x)
        Tnew = x.shape[2]
        
        # Convert to 2D for Swin Transformer
        x = threeD_to_2D_tensor(x)
        x = self.trunk(x, B, Tnew)
        
        # Add boundary information if used
        if self.use_boundary and boundaries is not None:
            x = torch.cat([x, boundaries], dim=-1)
        
        # Temporal modeling with DenseTCN
        return self.tcn(x, lengths, B)

    def _initialize_weights_randomly(self):
        """Initialize model weights randomly"""
        use_sqrt = True
        
        def f(n):
            return math.sqrt(2.0/float(n)) if use_sqrt else 2.0/float(n)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                # Calculate product of kernel size dimensions
                if isinstance(m.kernel_size, int):
                    kernel_prod = m.kernel_size
                else:
                    kernel_prod = 1
                    for k in m.kernel_size:
                        kernel_prod *= k
                n = kernel_prod * m.out_channels
                m.weight.data.normal_(0, f(n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, f(n))