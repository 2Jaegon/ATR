"""
Conditional Spatio-Temporal GNN Autoencoder (State-Aware) v2
=============================================================
문맥 변수 3개를 각각 독립 Embedding → concat → GNN 인코더 출력에 합산.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.transformer_ae import ContextEmbedding


class STGNNBlock(nn.Module):
    """Spatio-Temporal Graph Neural Network Block"""
    def __init__(self, num_features, hidden_dim):
        super(STGNNBlock, self).__init__()
        self.adj_matrix = nn.Parameter(torch.ones(num_features, num_features) / num_features)
        self.gcn_linear = nn.Linear(num_features, hidden_dim)
        self.tcn = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x_graph = torch.matmul(x, self.adj_matrix)
        x_spatial = F.relu(self.gcn_linear(x_graph))
        x_spatial_t = x_spatial.permute(0, 2, 1)
        x_temporal = F.relu(self.tcn(x_spatial_t))
        return x_temporal.permute(0, 2, 1)


class ConditionalSTGNNAE(nn.Module):
    """
    State-Aware 시공간 GNN 오토인코더
    - 인코더: GCN+TCN으로 시공간 특징 압축
    - 디코더: 인코딩 결과 + 문맥 임베딩 → 조건부 디코딩
    """
    def __init__(self, num_features=17,
                 num_recipe_steps=200, num_recipes=600, num_stages=400, num_tools=10,
                 embed_dim=8, hidden_dim=32):
        super(ConditionalSTGNNAE, self).__init__()

        self.context_embed = ContextEmbedding(num_recipe_steps, num_recipes, num_stages, num_tools, embed_dim)
        context_total_dim = embed_dim * 4

        self.encoder = STGNNBlock(num_features, hidden_dim)
        self.context_proj = nn.Linear(context_total_dim, hidden_dim)
        self.decoder = STGNNBlock(hidden_dim, num_features)
        self.output_proj = nn.Linear(num_features, num_features)

    def forward(self, x, context):
        ctx = self.context_embed(context)
        ctx_proj = self.context_proj(ctx)

        encoded = self.encoder(x)
        conditioned = encoded + ctx_proj
        decoded = self.decoder(conditioned)
        return self.output_proj(decoded)
