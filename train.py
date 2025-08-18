import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from dataset import LibriSpeechSpeakerDataset
from model import SpeakerNet
from loss import AAMSoftmax

def main(config):
    # --- 设置 ---
    device = torch.device(config['training']['device'])
    log_dir = config['logging']['log_dir']
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    
    # --- 数据加载 ---
    print("Loading datasets...")
    train_dataset = LibriSpeechSpeakerDataset(config, train=True)
    val_dataset = LibriSpeechSpeakerDataset(config, train=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['hardware']['num_workers'], pin_memory=config['hardware']['pin_memory'], drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
        num_workers=config['hardware']['num_workers'], pin_memory=config['hardware']['pin_memory']
    )
    
    # --- 模型、损失函数、优化器 ---
    print("Initializing model...")
    model = SpeakerNet(config).to(device)
    loss_fn = AAMSoftmax(config).to(device)
    
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=config['training']['learning_rate']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['training']['num_epochs'])
    
    use_amp = config['hardware']['use_amp']
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    best_val_loss = float('inf')
    
    # --- 训练循环 ---
    print("Starting training...")
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # --- 训练阶段 ---
        model.train()
        loss_fn.train()
        train_loss = []
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for mel_spec, speaker_id in train_progress:
            mel_spec, speaker_id = mel_spec.to(device), speaker_id.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                embedding = model(mel_spec)
                loss = loss_fn(embedding, speaker_id)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss.append(loss.item())
            train_progress.set_postfix(loss=np.mean(train_loss))

        avg_train_loss = np.mean(train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], epoch)
        
        # --- 验证阶段 ---
        model.eval()
        loss_fn.eval()
        val_loss = []
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for mel_spec, speaker_id in val_progress:
                mel_spec, speaker_id = mel_spec.to(device), speaker_id.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    embedding = model(mel_spec)
                    loss = loss_fn(embedding, speaker_id)
                val_loss.append(loss.item())
                val_progress.set_postfix(loss=np.mean(val_loss))

        avg_val_loss = np.mean(val_loss)
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        scheduler.step()
        
        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"Validation loss improved. Saved best model to {best_model_path}")

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)