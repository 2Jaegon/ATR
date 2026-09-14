import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    """
    LSTM 기반 오토인코더.
    직전 과거의 순차적(Sequential) 흐름을 기억하여 시계열 패턴을 복원합니다.
    """
    def __init__(self, num_features=17, hidden_dim=32, num_layers=1, dropout=0.0):
        super(LSTMAE, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 인코더: 시계열을 순차적으로 읽어 압축 (batch_first=True)
        self.encoder_lstm = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 디코더: 압축된 특징을 다시 원본 센서 개수(17)로 복원
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=num_features, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # 1. 인코딩 (압축)
        # lstm_out: (batch, seq_len, hidden_dim)
        enc_out, (h_n, c_n) = self.encoder_lstm(x)
        
        # 2. 디코딩 (복원)
        dec_out, _ = self.decoder_lstm(enc_out)
        
        return dec_out
