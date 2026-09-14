import torch
import torch.nn as nn

class SpatialGNNAE(nn.Module):
    """
    GNN 기반 오토인코더.
    17개 센서 간의 공간적/물리적 상호 인과관계를 그래프 형태로 학습합니다.
    """
    def __init__(self, num_features=17, hidden_dim=32):
        super(SpatialGNNAE, self).__init__()
        
        # 17개 센서 간의 연결 강도(Adjacency Matrix)를 모델 스스로 학습하도록 파라미터화
        self.adj_matrix = nn.Parameter(torch.ones(num_features, num_features) / num_features)
        
        # Graph Convolution 레이어 (단순화된 형태)
        self.gcn_encode = nn.Linear(num_features, hidden_dim)
        self.gcn_decode = nn.Linear(hidden_dim, num_features)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_features)
        # 1. 인접 행렬을 통한 센서 간 정보 교환 (메시지 패싱 흉내)
        x_graph = torch.matmul(x, self.adj_matrix)
        
        # 2. 특징 압축 (인코딩)
        hidden = torch.relu(self.gcn_encode(x_graph))
        
        # 3. 인접 행렬을 통한 복원 (디코딩)
        hidden_graph = torch.matmul(hidden, self.adj_matrix.T)
        out = self.gcn_decode(hidden_graph)
        
        return out
