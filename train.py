# train.py
import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import time

from dataset import LibriSpeechSpeakerDataset
from model import SpeakerNet
from loss import AAMSoftmax

def collate_fn(batch):
    """
    自定义 collate_fn，过滤掉值为 None 的样本。
    这对于处理太短或损坏的音频文件是必需的。
    """
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        # 如果整个批次都被过滤掉了，返回空的 tensor
        return torch.tensor([]), torch.tensor([])
    return default_collate(batch)

def main(config):
    # --- 设置 ---
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
    log_dir = config['logging']['log_dir']
    checkpoint_dir = config['logging']['checkpoint_dir']
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 为当前运行创建一个带时间戳的独立日志目录
    run_log_dir = os.path.join(log_dir, f"run_{int(time.time())}")
    writer = SummaryWriter(run_log_dir)
    print(f"TensorBoard logs for this run will be saved in: {run_log_dir}")
    use_amp = config['hardware']['use_amp']
    
    # --- 数据加载 ---
    print("Loading datasets...")
    # 从配置中读取是否应用扰动的标志
    apply_pert = config.get('perturbation', {}).get('enabled', False)
    if apply_pert:
        print("\n[!] Adversarial perturbation is ENABLED for the training set.\n")

    train_dataset = LibriSpeechSpeakerDataset(config, train=True, apply_perturbation=apply_pert)
    # 验证集永远不加扰动
    val_dataset = LibriSpeechSpeakerDataset(config, train=False, apply_perturbation=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config['training']['batch_size'], shuffle=True,
        num_workers=config['hardware']['num_workers'], pin_memory=config['hardware']['pin_memory'], drop_last=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['training']['batch_size'], shuffle=False,
        num_workers=config['hardware']['num_workers'], pin_memory=config['hardware']['pin_memory'],
        collate_fn=collate_fn
    )
    
    # --- 模型、损失函数、优化器 ---
    print("Initializing model...")
    num_speakers = len(train_dataset.speaker_to_id)
    model = SpeakerNet(config).to(device)
    # 将正确的说话人数量传递给损失函数
    loss_fn = AAMSoftmax(config, num_classes=num_speakers).to(device)
    
    # 将模型和损失函数的参数都交给优化器
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=config['training']['learning_rate']
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    
    # 自动混合精度 (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # --- 训练循环 ---
    print(f"Starting training on {device} with {num_speakers} speakers...")
    best_val_loss = float('inf')
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # --- 训练阶段 ---
        model.train()
        loss_fn.train()
        train_losses = []
        train_progress = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        for mel_spec, speaker_id in train_progress:
            # 跳过由 collate_fn 产生的空批次
            if mel_spec.nelement() == 0:
                continue
            
            mel_spec, speaker_id = mel_spec.to(device), speaker_id.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                embedding = model(mel_spec)
                loss = loss_fn(embedding, speaker_id)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            train_progress.set_postfix(loss=np.mean(train_losses))

        avg_train_loss = np.mean(train_losses) if train_losses else 0.0
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        
        # --- 验证阶段 ---
        model.eval()
        loss_fn.eval()
        val_losses = []
        val_progress = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.no_grad():
            for mel_spec, speaker_id in val_progress:
                # 跳过空批次
                if mel_spec.nelement() == 0:
                    continue
                
                mel_spec, speaker_id = mel_spec.to(device), speaker_id.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    embedding = model(mel_spec)
                    loss = loss_fn(embedding, speaker_id)
                
                val_losses.append(loss.item())
                val_progress.set_postfix(loss=np.mean(val_losses))

        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # --- 保存检查点 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            # 根据是否启用扰动，决定保存的文件名
            if config.get('perturbation', {}).get('enabled', False):
                model_filename = "best_model_perturbed.pt"
            else:
                model_filename = "best_model_clean.pt"

            checkpoint_path = os.path.join(checkpoint_dir, model_filename)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_fn_state_dict': loss_fn.state_dict(),
                'val_loss': avg_val_loss,
                'speaker_to_id': train_dataset.speaker_to_id
            }, checkpoint_path)
            print(f"Validation loss improved. Model saved to {checkpoint_path}")
            
        scheduler.step()

    writer.close()
    print("Training finished.")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)