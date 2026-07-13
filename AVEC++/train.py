import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import get_cosine_schedule_with_warmup
import torchaudio
import torchvision.io
import torchvision.transforms.functional as TF
import pandas as pd
import wandb

from model import AVEC_Hybrid, CTCTokenizer

# ==========================================
# 1. DATASET WITH 96x96 SWINLIP SIZING
# ==========================================
class LRSVoxMMDataset(Dataset):
    def __init__(self, manifest_path, tokenizer, max_frames=300):
        self.df = pd.read_csv(manifest_path)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.video_size = (96, 96) # Updated for SwinLip architecture

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Load Text
        text_target = torch.tensor(self.tokenizer.encode(row['reference']), dtype=torch.long)
        
        # 2. Load Audio
        audio_path = row['audio_path']
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0)
        
        # 3. Load Video
        video_path = row['video_path']
        video_frames, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
        
        # [T, H, W, C] -> [T, C, H, W]
        video_frames = video_frames.permute(0, 3, 1, 2).float() / 255.0
        
        # Interpolate dynamically treats T as the batch dimension here
        video_frames = F.interpolate(video_frames, size=self.video_size, mode='bilinear')
        
        # NOTE: If your SwinLip .pth was trained strictly on grayscale LRW datasets, 
        # uncomment the line below to convert the 3-channel RGB to 1-channel grayscale:
        video_frames = TF.rgb_to_grayscale(video_frames)
        
        return waveform, video_frames, text_target

def av_collate_fn(batch):
    waveforms, videos, targets = zip(*batch)
    
    audio_lengths = torch.tensor([w.size(0) for w in waveforms], dtype=torch.long)
    video_lengths = torch.tensor([v.size(0) for v in videos], dtype=torch.long)
    target_lengths = torch.tensor([t.size(0) for t in targets], dtype=torch.long)
    
    waveforms_padded = pad_sequence(waveforms, batch_first=True, padding_value=0.0)
    videos_padded = pad_sequence(videos, batch_first=True, padding_value=0.0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return waveforms_padded, audio_lengths, videos_padded, video_lengths, targets_padded, target_lengths

# ==========================================
# 2. VALIDATION LOOP
# ==========================================
def evaluate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_loss_final, val_loss_a, val_loss_v = 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for audio, a_lens, video, v_lens, targets, t_lens in dataloader:
            audio, a_lens = audio.to(device), a_lens.to(device)
            video, v_lens = video.to(device), v_lens.to(device)
            targets, t_lens = targets.to(device), t_lens.to(device)
            
            final_logits, a_inter, v_inter, out_lengths = model(audio, a_lens, video, v_lens)
            
            # [Batch, Time, Classes] -> [Time, Batch, Classes]
            final_logits = final_logits.transpose(0, 1).log_softmax(2)
            a_inter = a_inter.transpose(0, 1).log_softmax(2)
            v_inter = v_inter.transpose(0, 1).log_softmax(2)
            
            l_final = criterion(final_logits, targets, out_lengths, t_lens)
            l_a = criterion(a_inter, targets, out_lengths, t_lens)
            l_v = criterion(v_inter, targets, out_lengths, t_lens)
            
            total_loss = l_final + 0.1 * (l_a + l_v)
            
            val_loss += total_loss.item()
            val_loss_final += l_final.item()
            val_loss_a += l_a.item()
            val_loss_v += l_v.item()
            
    num_batches = len(dataloader)
    return (val_loss / num_batches, val_loss_final / num_batches, 
            val_loss_a / num_batches, val_loss_v / num_batches)

# ==========================================
# 3. TRAINING SCRIPT W/ WANDB
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Weights & Biases
    wandb.init(
        project="AVEC_LRS_VoxMM",
        name="swinlip-wav2vec2-hybrid",
        config={
            "epochs": 30,
            "batch_size": 4,
            "lr": 1e-4,
            "lambda_inter": 0.1,
            "video_size": 96,
            "grad_clip": 5.0
        }
    )
    config = wandb.config
    
    tokenizer = CTCTokenizer()
    
    # Ensure you have separate manifests for splits
    train_dataset = LRSVoxMMDataset("train_manifest.csv", tokenizer)
    dev_dataset = LRSVoxMMDataset("dev_manifest.csv", tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=av_collate_fn, num_workers=4)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=av_collate_fn, num_workers=4)
    
    model = AVEC_Hybrid(num_classes=len(tokenizer), swinlip_pth_path="path/to/swinlip.pth").to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=config.lr, weight_decay=1e-4)
    
    total_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.CTCLoss(blank=tokenizer.blank_idx, zero_infinity=True)
    
    # 2. Watch the model with wandb to log gradients and weights
    wandb.watch(model, criterion, log="all", log_freq=100)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (audio, a_lens, video, v_lens, targets, t_lens) in enumerate(train_loader):
            audio, a_lens = audio.to(device), a_lens.to(device)
            video, v_lens = video.to(device), v_lens.to(device)
            targets, t_lens = targets.to(device), t_lens.to(device)
            
            optimizer.zero_grad()
            final_logits, a_inter, v_inter, out_lengths = model(audio, a_lens, video, v_lens)
            
            final_logits = final_logits.transpose(0, 1).log_softmax(2)
            a_inter = a_inter.transpose(0, 1).log_softmax(2)
            v_inter = v_inter.transpose(0, 1).log_softmax(2)
            
            loss_final = criterion(final_logits, targets, out_lengths, t_lens)
            loss_audio = criterion(a_inter, targets, out_lengths, t_lens)
            loss_video = criterion(v_inter, targets, out_lengths, t_lens)
            
            loss = loss_final + config.lambda_inter * (loss_audio + loss_video)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            # 3. Log step-level metrics to wandb
            if batch_idx % 10 == 0:
                wandb.log({
                    "Train/Step_Loss": loss.item(),
                    "Train/Final_CTC": loss_final.item(),
                    "Train/Inter_Audio_CTC": loss_audio.item(),
                    "Train/Inter_Video_CTC": loss_video.item(),
                    "Train/Learning_Rate": scheduler.get_last_lr()[0]
                })

        avg_train_loss = train_loss / len(train_loader)
        
        # 4. Validation Phase
        print(f"Epoch {epoch+1} Complete. Running Validation on Dev Split...")
        val_loss, v_final, v_a, v_v = evaluate(model, dev_loader, criterion, device)
        
        # 5. Log Epoch-level metrics
        wandb.log({
            "Epoch": epoch + 1,
            "Train/Epoch_Loss": avg_train_loss,
            "Val/Total_Loss": val_loss,
            "Val/Final_CTC": v_final,
            "Val/Inter_Audio_CTC": v_a,
            "Val/Inter_Video_CTC": v_v
        })
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "avec_hybrid_best.pt")
            print("--> Saved New Best Model!")
            
            # Optionally log the artifact to wandb
            # wandb.save("avec_hybrid_best.pt")

    wandb.finish()

if __name__ == "__main__":
    train()