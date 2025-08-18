import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmax(nn.Module):
    def __init__(self, config):
        super().__init__()
        loss_config = config['training']['loss']
        self.in_features = config['model']['embedding_dim']
        self.out_features = config['training']['num_speakers']
        self.m = loss_config['margin']
        self.s = loss_config['scale']
        
        self.weight = nn.Parameter(torch.FloatTensor(self.out_features, self.in_features))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x, label):
        # L2 归一化 embedding 和权重
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        
        # 计算 phi
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 将 phi 应用到正确的类别上
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        # 计算交叉熵损失
        loss = F.cross_entropy(output, label)
        return loss