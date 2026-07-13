import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
from torchaudio.models import Conformer

# ==========================================
# TOKENIZER (CTC Vocabulary)
# ==========================================
class CTCTokenizer:
    def __init__(self):
        # Define character set based on typical English ASR (A-Z, space, apostrophe)
        self.chars = ["<blank>", "<unk>", " ", "'"] + [chr(i) for i in range(65, 91)]
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.blank_idx = self.char_to_idx["<blank>"]
        
    def __len__(self):
        return len(self.chars)

    def encode(self, text):
        return [self.char_to_idx.get(c, self.char_to_idx["<unk>"]) for c in text.upper()]

    def decode(self, indices):
        return "".join([self.idx_to_char.get(idx, "<unk>") for idx in indices if idx != self.blank_idx])

class AudioSubsampler(nn.Module):
    """Halves the audio feature sequence length using 1D Convolution."""
    def __init__(self, audio_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=audio_dim, 
            out_channels=audio_dim, 
            kernel_size=3, 
            stride=2, 
            padding=1
        )
        self.activation = nn.GELU()

    def forward(self, x, lengths):
        # x shape: [Batch, Time, Channels] -> [Batch, Channels, Time]
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = self.activation(x)
        x = x.transpose(1, 2)
        
        # Subsample the lengths tensor accordingly
        subsampled_lengths = ((lengths + 2 * 1 - 3) // 2) + 1
        return x, subsampled_lengths

class InterCTCBranch(nn.Module):
    """
    A specific modality branch (Audio or Video) containing Conformer blocks 
    and an intermediate CTC loss head halfway through.
    """
    def __init__(self, input_dim, conformer_dim, num_classes, num_layers=4, inter_layer=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, conformer_dim)
        
        # Bottom half of the conformers (before Intermediate CTC)
        self.conformer_bottom = Conformer(
            input_dim=conformer_dim, num_heads=4, ffn_dim=conformer_dim*4, 
            num_layers=inter_layer, depthwise_conv_kernel_size=31
        )
        
        # The Intermediate CTC Head
        self.inter_ctc_head = nn.Linear(conformer_dim, num_classes)
        
        # Top half of the conformers
        self.conformer_top = Conformer(
            input_dim=conformer_dim, num_heads=4, ffn_dim=conformer_dim*4, 
            num_layers=num_layers - inter_layer, depthwise_conv_kernel_size=31
        )

    def forward(self, x, lengths):
        x = self.proj(x)
        
        # Pass through bottom layers
        x, lengths = self.conformer_bottom(x, lengths)
        
        # Extract intermediate predictions
        inter_logits = self.inter_ctc_head(x)
        
        # Pass through top layers (Inter-CTC residual concept can also be added here)
        x, lengths = self.conformer_top(x, lengths)
        
        return x, inter_logits, lengths

class DummySwinLip(nn.Module):
    """Placeholder for your SwinLip .pth model."""
    def __init__(self, output_dim=512):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, video_frames):
        batch_size, time_v = video_frames.shape[0], video_frames.shape[1]
        return torch.randn(batch_size, time_v, self.output_dim, device=video_frames.device)

class AVEC_Hybrid(nn.Module):
    def __init__(self, num_classes, swinlip_pth_path=None):
        super().__init__()
        
        # ==========================================
        # 1. ENCODERS (To be frozen)
        # ==========================================
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        audio_dim = self.audio_encoder.config.hidden_size # 768
        
        self.video_encoder = DummySwinLip(output_dim=512)
        if swinlip_pth_path:
            # Load your SwinLip weights from Google Drive here
            # self.video_encoder.load_state_dict(torch.load(swinlip_pth_path))
            pass
        video_dim = 512
        
        # ==========================================
        # 2. MODALITY BRANCHES (Trainable)
        # ==========================================
        conformer_dim = 512
        
        self.audio_subsampler = AudioSubsampler(audio_dim)
        
        # Audio Conformer with Inter-CTC
        self.audio_branch = InterCTCBranch(
            input_dim=audio_dim, conformer_dim=conformer_dim, 
            num_classes=num_classes, num_layers=4, inter_layer=2
        )
        
        # Video Conformer with Inter-CTC
        self.video_branch = InterCTCBranch(
            input_dim=video_dim, conformer_dim=conformer_dim, 
            num_classes=num_classes, num_layers=4, inter_layer=2
        )
        
        # ==========================================
        # 3. FUSION & JOINT NETWORK (Trainable)
        # ==========================================
        self.fusion_proj = nn.Linear(conformer_dim * 2, conformer_dim)
        
        self.joint_conformer = Conformer(
            input_dim=conformer_dim, num_heads=4, ffn_dim=conformer_dim*4, 
            num_layers=2, depthwise_conv_kernel_size=31
        )
        
        self.final_ctc_head = nn.Linear(conformer_dim, num_classes)
        
        # Freeze the encoders upon initialization
        self._freeze_encoders()

    def _freeze_encoders(self):
        """Freezes Wav2Vec2 and SwinLip so only the back-end is trained."""
        for param in self.audio_encoder.parameters():
            param.requires_grad = False
        for param in self.video_encoder.parameters():
            param.requires_grad = False
            
    def forward(self, audio_waveform, audio_lengths, video_frames, video_lengths):
        
        # --- AUDIO MODALITY ---
        with torch.no_grad(): # Ensure no gradients are tracked for the encoder
            audio_out = self.audio_encoder(audio_waveform).last_hidden_state
            
        # Subsample to 25Hz and pass through audio conformer blocks
        a_feat, a_lengths = self.audio_subsampler(audio_out, audio_lengths)
        a_feat, a_inter_logits, a_lengths = self.audio_branch(a_feat, a_lengths)
        
        # --- VIDEO MODALITY ---
        with torch.no_grad():
            v_feat = self.video_encoder(video_frames)
            
        v_feat, v_inter_logits, v_lengths = self.video_branch(v_feat, video_lengths)
        
        # --- ALIGNMENT ---
        # Force sequences to match the shortest temporal dimension
        min_len = min(a_feat.shape[1], v_feat.shape[1])
        a_feat = a_feat[:, :min_len, :]
        v_feat = v_feat[:, :min_len, :]
        
        # Truncate logits similarly so CTC loss dimensions match the targets
        a_inter_logits = a_inter_logits[:, :min_len, :]
        v_inter_logits = v_inter_logits[:, :min_len, :]
        
        # Create a unified length tensor for the joint conformer
        joint_lengths = torch.clamp(a_lengths, max=min_len) 
        
        # --- FUSION ---
        # Concatenate Audio and Video features
        fused = torch.cat((a_feat, v_feat), dim=-1)
        fused = self.fusion_proj(fused)
        
        # --- JOINT CONFORMER ---
        joint_feat, joint_lengths = self.joint_conformer(fused, joint_lengths)
        final_logits = self.final_ctc_head(joint_feat)
        
        return final_logits, a_inter_logits, v_inter_logits, joint_lengths