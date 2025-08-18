import torch
import torch.nn as nn
from torchaudio.models import Conformer

class SpeakerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        
        self.conformer = Conformer(
            input_dim=model_config['input_dim'],
            num_heads=model_config['num_heads'],
            ffn_dim=model_config['ffn_dim'],
            num_layers=model_config['num_layers'],
            depthwise_conv_kernel_size=model_config['depthwise_conv_kernel_size'],
        )
        
        # Conformer的输出维度是ffn_dim，我们需要将其映射到我们想要的embedding维度
        self.embedding_head = nn.Sequential(
            # 简单地对时间维度做平均池化
            # 然后通过一个线性层得到最终的embedding
            nn.Linear(model_config['ffn_dim'], model_config['embedding_dim'])
        )

    def forward(self, x):
        # Conformer 输入需要 (batch, time, freq)
        # 我们的 Mel 频谱是 (batch, time, freq)，正好匹配
        # Conformer 同时需要一个表示长度的张量
        lengths = torch.full((x.size(0),), x.size(1), device=x.device)
        
        x, _ = self.conformer(x, lengths)
        
        # 对时间维度进行平均池化
        x = torch.mean(x, dim=1)
        
        # 通过线性头得到 embedding
        embedding = self.embedding_head(x)
        
        return embedding