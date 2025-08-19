import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AAMSoftmax(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        model_config = config['model']
        loss_config = config['loss']
        
        self.embedding_dim = model_config['embedding_dim']
        self.num_classes = num_classes
        self.margin = loss_config['margin']
        self.scale = loss_config['scale']

        # 分类权重
        self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, self.embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # 1. 归一化 embedding 和分类权重
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # 2. 计算 cosine 相似度 (logits)
        cosine = torch.matmul(embeddings_norm, weight_norm.t())
        
        # 3. AAM (Additive Angular Margin)
        # 将 cosine 转换为角度 theta
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        
        # 为目标类别加上角度 margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        target_theta = theta + self.margin
        
        # 将加了 margin 的角度转换回 cosine
        target_cosine = torch.cos(target_theta)
        
        # 组合修改后的 logits
        output = (one_hot * target_cosine) + ((1.0 - one_hot) * cosine)
        
        # 4. 缩放 logits
        output *= self.scale

        # 5. 计算交叉熵损失
        loss = F.cross_entropy(output, labels)
        return loss