import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer, TransformerEncoder

class SpeakerNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        self.n_mels = model_config['n_mels']
        self.embedding_dim = model_config['embedding_dim']
        
        # Conformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=model_config['d_model'],
            nhead=model_config['n_heads'],
            dim_feedforward=model_config['d_model'] * model_config['ff_expansion_factor'],
            dropout=model_config['dropout'],
            activation='relu',
            batch_first=True
        )
        self.conformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=model_config['n_layers']
        )
        
        # Input projection to match d_model
        self.input_proj = nn.Linear(self.n_mels, model_config['d_model'])
        
        # Attentive Statistics Pooling
        self.asp_tdnn = nn.Conv1d(in_channels=model_config['d_model'], out_channels=model_config['d_model'], kernel_size=1)
        self.asp_attention = nn.Conv1d(in_channels=model_config['d_model'], out_channels=1, kernel_size=1)
        
        # Final projection to embedding space
        self.final_proj = nn.Linear(model_config['d_model'] * 2, self.embedding_dim)

    def forward(self, x):
        # Input shape: (batch, 1, n_mels, time) -> (batch, n_mels, time)
        x = x.squeeze(1)
        # (batch, n_mels, time) -> (batch, time, n_mels)
        x = x.transpose(1, 2)
        
        # Project input to d_model
        # (batch, time, n_mels) -> (batch, time, d_model)
        x = self.input_proj(x)
        
        # Conformer Encoder
        # (batch, time, d_model) -> (batch, time, d_model)
        x = self.conformer_encoder(x)
        
        # Attentive Statistics Pooling
        # (batch, time, d_model) -> (batch, d_model, time)
        x = x.transpose(1, 2)
        
        h = torch.tanh(self.asp_tdnn(x))
        w = torch.softmax(self.asp_attention(h), dim=2)
        
        mu = torch.sum(x * w, dim=2)
        sigma = torch.sqrt(torch.sum((x**2) * w, dim=2) - mu**2)
        
        # Concatenate mean and std
        # (batch, d_model * 2)
        x = torch.cat((mu, sigma), dim=1)
        
        # Final projection
        # (batch, d_model * 2) -> (batch, embedding_dim)
        x = self.final_proj(x)
        
        # L2 Normalization
        x = F.normalize(x, p=2, dim=1)
        
        return x