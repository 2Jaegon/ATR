import torch
import torch.nn as nn

class TransformerAE(nn.Module):
    """
    SOTA 시계열 이상 탐지를 위한 트랜스포머 기반 오토인코더(Autoencoder)
    """
    def __init__(self, num_features=17, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerAE, self).__init__()
        
        # 입력 차원 17을 트랜스포머 d_model 차원으로 투영
        self.encoder_proj = nn.Linear(num_features, d_model)
        
        # Self-Attention 기반 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # d_model 차원을 다시 원본 센서 차원(17)으로 디코딩(복원)
        self.decoder_proj = nn.Linear(d_model, num_features)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        h = self.encoder_proj(x)
        h = self.transformer_encoder(h)
        out = self.decoder_proj(h)
        return out
