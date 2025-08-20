import torch
import torchaudio
import os
import yaml

# 导入我们自己的模型定义
from model import SpeakerNet

class Perturbation:
    def __init__(self, config):
        """
        初始化扰动生成器。
        这个类现在会加载一个我们自己训练的 SpeakerNet 模型作为代理模型。
        """
        pert_config = config['perturbation']
        audio_config = config['audio']
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
        
        print("\n" + "="*50)
        print("Initializing Perturbation Module (Self-Adversarial Mode)...")
        
        # 1. 加载我们自己的 SpeakerNet 模型作为代理模型
        print("Loading our own SpeakerNet as the surrogate model...")
        self.surrogate_model = SpeakerNet(config).to(self.device)
        
        # 加载检查点
        checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model_clean.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found at {checkpoint_path}. Please train a model on the clean dataset first.")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.surrogate_model.load_state_dict(checkpoint['model_state_dict'])
        self.surrogate_model.eval()
        print(f"Surrogate model loaded from {checkpoint_path}.")
        
        # 2. 初始化 Mel 频谱图转换器，因为我们的模型需要它
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=audio_config['sample_rate'],
            n_fft=audio_config['n_fft'],
            win_length=audio_config['win_length'],
            hop_length=audio_config['hop_length'],
            n_mels=audio_config['n_mels']
        )

        # 3. PGD 攻击参数
        self.epsilon = pert_config['epsilon']
        self.alpha = pert_config['alpha']
        self.num_iter = pert_config['num_iter']
        print(f"PGD params: epsilon={self.epsilon}, alpha={self.alpha}, num_iter={self.num_iter}")
        print("="*50 + "\n")

    def to_mel_spec(self, waveform):
        """Helper function to convert waveform to log-mel-spectrogram."""
        # Ensure the transform is on the same device as the waveform
        mel_spec = self.mel_transform.to(waveform.device)(waveform)
        return torch.log(mel_spec + 1e-9)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        对输入的音频波形应用PGD扰动。
        :param waveform: 输入的音频波形张量，形状为 (1, 采样点数)
        :return: 经过扰动后的音频波形张量
        """
        waveform = waveform.to(self.device)
        waveform.requires_grad = False # Ensure original waveform does not require grad

        # 1. 计算原始声纹嵌入作为目标"锚点"
        with torch.no_grad():
            original_spec = self.to_mel_spec(waveform)
            original_embedding = self.surrogate_model(original_spec).detach()
        
        # 2. 初始化扰动 delta
        delta = torch.zeros_like(waveform, requires_grad=True)
        
        # 3. PGD 迭代
        for _ in range(self.num_iter):
            perturbed_audio = waveform + delta
            
            # 将扰动后的音频转换为频谱图以输入模型
            perturbed_spec = self.to_mel_spec(perturbed_audio)
            perturbed_embedding = self.surrogate_model(perturbed_spec)
            
            # 损失函数：最小化负的余弦相似度（即最大化差异）
            loss = -torch.nn.functional.cosine_similarity(perturbed_embedding, original_embedding).mean()
            
            # 反向传播计算梯度 (梯度会从模型->频谱图->一直传到 delta)
            loss.backward()
            
            with torch.no_grad():
                # 获取梯度符号
                grad_sign = delta.grad.data.sign()
                # 更新 delta
                delta.data = delta.data - self.alpha * grad_sign
                # 投影: 保证扰动大小在 epsilon 范围内
                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)
                # 保证添加扰动后的音频值在有效范围内 [-1, 1]
                clamped_perturbed_audio = torch.clamp(waveform + delta.data, -1.0, 1.0)
                # 更新 delta 以反映对音频的裁剪
                delta.data = clamped_perturbed_audio - waveform.data

            # 清除梯度
            delta.grad.data.zero_()

        # 返回在原始设备上的扰动后音频
        return (waveform + delta).detach().to(waveform.device)
