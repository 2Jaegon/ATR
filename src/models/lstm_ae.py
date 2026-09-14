import torch
import torch.nn as nn

class LSTMAE(nn.Module):
    """
    정통 논문 수준의 Seq2Seq LSTM-AE (Encoder-Decoder 구조).
    전체 시퀀스를 단일 벡터(Context Vector)로 압축한 뒤 원본 데이터를 완벽히 복원합니다.
    (Malhotra et al., 2016 기반)
    """
    def __init__(self, num_features=17, hidden_dim=32, num_layers=1, dropout=0.0):
        super(LSTMAE, self).__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        
        # Encoder: 시퀀스를 읽어 단일 컨텍스트 벡터(h_n)로 압축
        self.encoder = nn.LSTM(
            input_size=num_features, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder: 압축된 컨텍스트 벡터를 기반으로 시퀀스를 펼침
        # 디코더의 입력은 압축된 벡터 자체가 되므로 input_size = hidden_dim
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 복원 층: 은닉 상태를 원본 센서 개수(17)로 투영
        self.output_layer = nn.Linear(hidden_dim, num_features)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        batch_size, seq_len, _ = x.size()
        
        # 1. 인코딩 (압축)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        _, (h_n, c_n) = self.encoder(x)
        
        # 컨텍스트 벡터: 가장 마지막 레이어의 은닉 상태를 가져옴
        context_vector = h_n[-1] # shape: (batch_size, hidden_dim)
        
        # 2. 벡터 복제 (Repeat)
        # 압축된 1개의 벡터를 원래 시계열 길이(seq_len)만큼 복제하여 디코더의 입력으로 사용
        # shape: (batch_size, seq_len, hidden_dim)
        decoder_input = context_vector.unsqueeze(1).repeat(1, seq_len, 1)
        
        # 3. 디코딩 (복원)
        # 초기 은닉 상태(h_n, c_n)를 인코더에서 그대로 물려받아 디코딩 시작
        decoder_out, _ = self.decoder(decoder_input, (h_n, c_n))
        
        # 4. 최종 출력 매핑
        out = self.output_layer(decoder_out)
        
        return out
