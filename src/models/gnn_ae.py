import torch
import torch.nn as nn
import torch.nn.functional as F

class STGNNBlock(nn.Module):
    """
    Spatio-Temporal Graph Neural Network Block
    (공간적 특징과 시간적 특징을 동시에 추출하는 핵심 블록)
    """
    def __init__(self, num_features, hidden_dim):
        super(STGNNBlock, self).__init__()
        
        # 1. Spatial GCN (공간 인과관계 추출)
        self.adj_matrix = nn.Parameter(torch.ones(num_features, num_features) / num_features)
        self.gcn_linear = nn.Linear(num_features, hidden_dim)
        
        # 2. Temporal CNN (시간 흐름 추출: 커널 크기 3의 1D-CNN)
        # 패딩을 통해 시간 길이(seq_len)를 보존
        self.tcn = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        batch_size, seq_len, num_features = x.size()
        
        # --- Spatial GCN ---
        # 인접 행렬을 통한 노드(센서) 간 메시지 패싱
        # (batch_size, seq_len, num_features) * (num_features, num_features) -> (batch_size, seq_len, num_features)
        x_graph = torch.matmul(x, self.adj_matrix)
        x_spatial = F.relu(self.gcn_linear(x_graph)) # (batch_size, seq_len, hidden_dim)
        
        # --- Temporal CNN ---
        # Conv1d는 (batch_size, channels, seq_len) 형태를 요구함
        x_spatial_t = x_spatial.permute(0, 2, 1) # (batch_size, hidden_dim, seq_len)
        x_temporal = F.relu(self.tcn(x_spatial_t)) # (batch_size, hidden_dim, seq_len)
        
        # 원상 복구
        out = x_temporal.permute(0, 2, 1) # (batch_size, seq_len, hidden_dim)
        return out

class SpatialGNNAE(nn.Module):
    """
    정통 논문 수준의 시공간(Spatio-Temporal) GNN 오토인코더
    """
    def __init__(self, num_features=17, hidden_dim=32):
        super(SpatialGNNAE, self).__init__()
        
        # 인코더: ST-GNN 블록을 거쳐 시공간 특징 압축
        self.encoder = STGNNBlock(num_features, hidden_dim)
        
        # 디코더: ST-GNN 블록을 역으로 거쳐 원본 데이터 차원으로 복원
        self.decoder = STGNNBlock(hidden_dim, num_features)
        
        # 출력 스케일링을 위한 최종 투영
        self.output_proj = nn.Linear(num_features, num_features)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        
        # 인코딩 (특징 압축)
        encoded = self.encoder(x)
        
        # 디코딩 (시공간 특징 복원)
        decoded = self.decoder(encoded)
        
        # 최종 출력
        out = self.output_proj(decoded)
        return out
