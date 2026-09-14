import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    시계열 데이터의 시간 순서를 기억하게 하는 위치 인코딩 (Sinusoidal PE)
    Attention Is All You Need (Vaswani et al., 2017)
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # batch_first=True 이므로 (1, max_len, d_model)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        device = x.device
        x = x + self.pe[:, :x.size(1), :].to(device)
        return x

class TransformerAE(nn.Module):
    """
    SOTA 시계열 이상 탐지를 위한 트랜스포머 기반 오토인코더
    (Positional Encoding 및 정통 Encoder-Decoder 구조 반영)
    """
    def __init__(self, num_features=17, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerAE, self).__init__()
        
        self.d_model = d_model
        
        # 1. 입력 투영 및 위치 인코딩
        self.encoder_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # 2. Transformer Encoder (특징 압축)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 3. Transformer Decoder (특징 복원)
        # 자기 자신을 타겟으로 하여 시계열 재구성 (Auto-regressive 또는 마스킹 없이 단순 복원)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # 4. 출력 투영 (d_model -> 원본 차원 17)
        self.decoder_proj = nn.Linear(d_model, num_features)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        
        # 임베딩 & 위치 인코딩
        h = self.encoder_proj(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)
        
        # 인코더: 시계열 전체의 컨텍스트(Attention) 압축
        memory = self.transformer_encoder(h)
        
        # 디코더: 인코딩된 memory를 바탕으로 원래의 입력 형태(h)를 타겟으로 복원
        # (이상 탐지를 위한 오토인코더에서는 타겟과 입력을 동일하게 주고 복원 오차를 계산)
        out_emb = self.transformer_decoder(tgt=h, memory=memory)
        
        # 최종 복원된 원본 데이터 차원
        out = self.decoder_proj(out_emb)
        return out
