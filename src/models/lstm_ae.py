"""
Conditional LSTM Autoencoder (State-Aware) v2
==============================================
문맥 변수 3개를 각각 독립 Embedding → concat → 인코더 h_n과 융합 → 조건부 디코딩.
"""
import torch
import torch.nn as nn
from src.models.transformer_ae import ContextEmbedding


class ConditionalLSTMAE(nn.Module):
    """
    State-Aware LSTM 오토인코더
    - 인코더: 시퀀스를 단일 컨텍스트 벡터(h_n)로 압축
    - 디코더: h_n + 문맥 임베딩을 concat → FC로 융합 → 조건부 디코딩
    """
    def __init__(self, num_features=17,
                 num_recipe_steps=200, num_recipes=600, num_stages=400, num_tools=10,
                 embed_dim=8, hidden_dim=32, num_layers=1, dropout=0.0):
        super(ConditionalLSTMAE, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.context_embed = ContextEmbedding(num_recipe_steps, num_recipes, num_stages, num_tools, embed_dim)
        context_total_dim = embed_dim * 4

        self.encoder = nn.LSTM(
            input_size=num_features, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.context_fusion = nn.Linear(hidden_dim + context_total_dim, hidden_dim)

        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_layer = nn.Linear(hidden_dim, num_features)

    def forward(self, x, context):
        batch_size, seq_len, _ = x.size()

        ctx = self.context_embed(context)   # (B, S, 3*embed_dim)
        ctx_pooled = ctx.mean(dim=1)        # (B, 3*embed_dim)

        _, (h_n, c_n) = self.encoder(x)
        encoder_hidden = h_n[-1]            # (B, hidden_dim)

        fused = torch.cat([encoder_hidden, ctx_pooled], dim=-1)
        fused = torch.relu(self.context_fusion(fused))

        decoder_input = fused.unsqueeze(1).repeat(1, seq_len, 1)
        h_0 = fused.unsqueeze(0).repeat(self.num_layers, 1, 1)
        decoder_out, _ = self.decoder(decoder_input, (h_0, c_n))

        return self.output_layer(decoder_out)
