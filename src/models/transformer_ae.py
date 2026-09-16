"""
Conditional Transformer-based Autoencoder (State-Aware) v2
==========================================================
문맥 변수 3개(recipe_step, recipe, stage)를 각각 독립적인 Embedding Layer로 처리.
고카디널리티(164/527/352종)에도 메모리 효율적으로 대응합니다.
"""
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """시계열 데이터의 시간 순서를 기억하게 하는 위치 인코딩 (Sinusoidal PE)"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        device = x.device
        x = x + self.pe[:, :x.size(1), :].to(device)
        return x


class ContextEmbedding(nn.Module):
    """
    4개의 문맥 변수(recipe_step, recipe, stage, Tool)를 각각 Embedding → concat.
    context shape: (B, S, 4) with integer indices
    output shape: (B, S, total_embed_dim)
    """
    def __init__(self, num_recipe_steps, num_recipes, num_stages, num_tools, embed_dim=8):
        super(ContextEmbedding, self).__init__()
        self.embed_rs = nn.Embedding(num_recipe_steps, embed_dim)
        self.embed_r = nn.Embedding(num_recipes, embed_dim)
        self.embed_s = nn.Embedding(num_stages, embed_dim)
        self.embed_t = nn.Embedding(num_tools, embed_dim)
        self.total_dim = embed_dim * 4

    def forward(self, context):
        # context: (B, S, 4) — [recipe_step_idx, recipe_idx, stage_idx, tool_idx]
        rs_emb = self.embed_rs(context[:, :, 0])  # (B, S, embed_dim)
        r_emb = self.embed_r(context[:, :, 1])
        s_emb = self.embed_s(context[:, :, 2])
        t_emb = self.embed_t(context[:, :, 3])
        return torch.cat([rs_emb, r_emb, s_emb, t_emb], dim=-1)  # (B, S, 4*embed_dim)


class ConditionalTransformerAE(nn.Module):
    """
    State-Aware Transformer 오토인코더
    - 인코더: Self-Attention으로 시계열 장기 의존성 압축
    - 디코더: 인코딩된 memory에 문맥 벡터를 조건으로 주입하여 복원
    """
    def __init__(self, num_features=17,
                 num_recipe_steps=200, num_recipes=600, num_stages=400, num_tools=10,
                 embed_dim=8, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(ConditionalTransformerAE, self).__init__()

        self.d_model = d_model

        # 문맥 임베딩 (4개 변수 → 4*embed_dim 차원)
        self.context_embed = ContextEmbedding(num_recipe_steps, num_recipes, num_stages, num_tools, embed_dim)
        context_total_dim = embed_dim * 4

        # 입력 투영 및 위치 인코딩
        self.encoder_proj = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 문맥 조건 투영 (context_total_dim → d_model)
        self.context_proj = nn.Linear(context_total_dim, d_model)

        # Transformer Decoder (조건부 복원)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # 출력 투영
        self.decoder_proj = nn.Linear(d_model, num_features)

    def forward(self, x, context):
        """
        x: (B, S, num_features) — 센서 데이터
        context: (B, S, 4) — 정수 인덱스 [recipe_step, recipe, stage, Tool]
        """
        ctx = self.context_embed(context)       # (B, S, 4*embed_dim)
        ctx_proj = self.context_proj(ctx)        # (B, S, d_model)

        h = self.encoder_proj(x) * math.sqrt(self.d_model)
        h = self.pos_encoder(h)

        memory = self.transformer_encoder(h)
        conditioned_memory = memory + ctx_proj

        out_emb = self.transformer_decoder(tgt=h, memory=conditioned_memory)
        out = self.decoder_proj(out_emb)
        return out
