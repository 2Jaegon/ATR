"""
State-Aware 동적 앙상블 탐지 에이전트
=====================================
- Conditional 모델 3종 (Transformer, ST-GNN, LSTM)의 실시간 추론
- 공정 스텝(recipe_step) 기반 동적 가중합 (Dynamic Weighted Ensemble)
- POT 동적 임계값 적용 (극단값 이론 기반 오경보 최소화)
"""
import numpy as np
import os
import joblib
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from src.models.transformer_ae import ConditionalTransformerAE
    from src.models.gnn_ae import ConditionalSTGNNAE
    from src.models.lstm_ae import ConditionalLSTMAE
    TORCH_AVAILABLE = True
except OSError as e:
    print(f"[Warning] PyTorch DLL Error in DetectionAgent ({e}). Mock Inference Mode activated.")
    TORCH_AVAILABLE = False

from src.data.preprocessing import (
    PreprocessingPipeline, SENSOR_FEATURES, CONTEXT_FEATURES
)


class DetectionAgent:
    """
    State-Aware Dynamic Ensemble Detection Agent
    - 문맥 변수(recipe_step, stage 등)를 인지하는 Conditional 모델 추론
    - 공정 스텝별 동적 가중합 앙상블
    - POT 기반 동적 임계값
    """

    def __init__(self, seq_len=5, weights_dir="src/models/weights"):
        self.seq_len = seq_len
        self.weights_dir = weights_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if TORCH_AVAILABLE else None

        self.model_tr = None
        self.model_gnn = None
        self.model_lstm = None

        # 전처리 파이프라인
        self.pipeline = PreprocessingPipeline(weights_dir=weights_dir)

        # 동적 가중치 (기본값)
        self.default_weights = {'TR': 0.33, 'GNN': 0.34, 'LSTM': 0.33}

        # POT 임계값 (기본값, 로드 시 덮어씀)
        self.yellow_threshold = 0.6
        self.red_threshold = 0.8
        self.pot_method = 'fixed'

        self.is_loaded = False

    # ─── 동적 가중치 로직 ─────────────────────────────────
    def _get_dynamic_weights(self, recipe_step):
        """
        공정 스텝에 따라 모델별 가중치를 동적으로 조절.
        
        - 동적 스텝 (가스 주입, 밸브 개폐, 초기 셋업):
          → 센서 간 상호작용이 급변 → ST-GNN 비중 ↑
        - 안정 유지 스텝 (에칭 진행, 장기 안정):
          → 순차적 흐름/장기 의존성 중요 → Transformer/LSTM 비중 ↑
        """
        try:
            step = int(recipe_step)
        except (ValueError, TypeError):
            return self.default_weights

        # 초기 셋업 / 가스 전환 / 밸브 동작 스텝
        dynamic_steps = {1, 2, 3, 5, 6}
        # 안정 에칭 / 장기 유지 스텝
        stable_steps = {7, 8, 9, 10, 11, 12, 13, 14, 15}

        if step in dynamic_steps:
            return {'TR': 0.20, 'GNN': 0.50, 'LSTM': 0.30}
        elif step in stable_steps:
            return {'TR': 0.40, 'GNN': 0.20, 'LSTM': 0.40}
        else:
            return self.default_weights

    # ─── 가중치/모델 로드 ─────────────────────────────────
    def load_weights(self):
        """학습된 모델 가중치, 전처리 파이프라인, POT 임계값 로드"""

        # 1) 전처리 파이프라인 로드
        self.pipeline.load()

        # 2) POT 임계값 로드
        pot_path = os.path.join(self.weights_dir, "pot_threshold.pkl")
        if os.path.exists(pot_path):
            pot = joblib.load(pot_path)
            self.yellow_threshold = pot.get('yellow_threshold', 0.6)
            self.red_threshold = pot.get('red_threshold', 0.8)
            self.pot_method = pot.get('method', 'fixed')
            print(f"[DetectionAgent] POT 임계값 로드: Yellow={self.yellow_threshold:.6f}, Red={self.red_threshold:.6f} ({self.pot_method})")
        else:
            print("[DetectionAgent] POT 파일 없음. 고정 임계값 사용 (0.6 / 0.8)")

        # 3) PyTorch 모델 로드
        if not TORCH_AVAILABLE:
            print("[DetectionAgent] Mock Inference 모드로 준비되었습니다.")
            self.is_loaded = True
            return

        num_features = len(SENSOR_FEATURES)
        num_recipe_steps = self.pipeline.num_recipe_steps
        num_recipes = self.pipeline.num_recipes
        num_stages = self.pipeline.num_stages
        num_tools = self.pipeline.num_tools

        self.model_tr = ConditionalTransformerAE(
            num_features=num_features,
            num_recipe_steps=num_recipe_steps, num_recipes=num_recipes, num_stages=num_stages, num_tools=num_tools,
            embed_dim=8, d_model=32, nhead=4, num_layers=2
        ).to(self.device)

        self.model_gnn = ConditionalSTGNNAE(
            num_features=num_features,
            num_recipe_steps=num_recipe_steps, num_recipes=num_recipes, num_stages=num_stages, num_tools=num_tools,
            embed_dim=8, hidden_dim=16
        ).to(self.device)

        self.model_lstm = ConditionalLSTMAE(
            num_features=num_features,
            num_recipe_steps=num_recipe_steps, num_recipes=num_recipes, num_stages=num_stages, num_tools=num_tools,
            embed_dim=8, hidden_dim=32, num_layers=2
        ).to(self.device)

        loaded = []
        for name, model, fname in [
            ("TR-AE", self.model_tr, "tr_ae.pt"),
            ("ST-GNN", self.model_gnn, "st_gnn.pt"),
            ("LSTM-AE", self.model_lstm, "lstm_ae.pt")
        ]:
            path = os.path.join(self.weights_dir, fname)
            if os.path.exists(path):
                try:
                    model.load_state_dict(torch.load(path, map_location=self.device))
                    loaded.append(name)
                except Exception as e:
                    print(f"[DetectionAgent] {name} 로드 에러: {e}")
            else:
                print(f"[DetectionAgent] {name} 가중치 없음 ({fname}) - 학습 대기 상태")

        print(f"[DetectionAgent] PyTorch 가중치 로드 완료: {len(loaded)}/3 ({', '.join(loaded)})")

        self.is_loaded = True

    # ─── 추론 ─────────────────────────────────────────────
    def detect(self, current_data: pd.DataFrame) -> dict:
        """
        스트리밍 데이터 청크(최소 seq_len)를 받아 이상 점수를 산출합니다.

        Args:
            current_data: DataFrame (센서 + 문맥 변수 컬럼 포함)
        Returns:
            dict: {status, final_score, scores, weights, recipe_step}
        """
        # 현재 recipe_step 추출 (동적 가중치용)
        current_recipe_step = None
        if 'recipe_step' in current_data.columns:
            current_recipe_step = current_data['recipe_step'].iloc[-1]

        # 동적 가중치 결정
        weights = self._get_dynamic_weights(current_recipe_step)

        # 전처리: 센서 스케일링 + 문맥 인코딩
        if self.pipeline._is_fitted:
            sensor_scaled, context_vec = self.pipeline.transform(current_data)
        else:
            available = [f for f in SENSOR_FEATURES if f in current_data.columns]
            sensor_scaled = current_data[available].values.astype(np.float32)
            context_vec = np.zeros((len(current_data), 4), dtype=np.int64)

        # 윈도우 생성
        sensor_tensor = torch.FloatTensor(sensor_scaled).unsqueeze(0)   # (1, S, F)
        context_tensor = torch.LongTensor(context_vec).unsqueeze(0)     # (1, S, 4)

        # 각 모델별 Reconstruction Error
        if TORCH_AVAILABLE and self.model_tr is not None:
            tr_loss = self._infer(self.model_tr, sensor_tensor, context_tensor)
            gnn_loss = self._infer(self.model_gnn, sensor_tensor, context_tensor)
            lstm_loss = self._infer(self.model_lstm, sensor_tensor, context_tensor)
        else:
            # Mock Inference
            var_score = np.var(sensor_scaled) * 10
            tr_loss = float(np.random.uniform(0.1, 0.3) + var_score)
            gnn_loss = float(np.random.uniform(0.1, 0.3) + var_score)
            lstm_loss = float(np.random.uniform(0.1, 0.3) + var_score)

        # 동적 가중합 앙상블
        final_score = (
            weights['TR'] * tr_loss +
            weights['GNN'] * gnn_loss +
            weights['LSTM'] * lstm_loss
        )

        # POT 동적 임계값 기반 상태 판정
        status = "normal"
        if final_score > self.red_threshold:
            status = "red"
        elif final_score > self.yellow_threshold:
            status = "yellow"

        return {
            "status": status,
            "final_score": float(final_score),
            "scores": {
                "TR": float(tr_loss),
                "GNN": float(gnn_loss),
                "LSTM": float(lstm_loss)
            },
            "weights": weights,
            "recipe_step": int(current_recipe_step) if current_recipe_step is not None else None,
            "thresholds": {
                "yellow": self.yellow_threshold,
                "red": self.red_threshold,
                "method": self.pot_method
            }
        }

    def _infer(self, model, sensor_tensor, context_tensor):
        """단일 모델 추론 (Reconstruction Error 산출)"""
        model.eval()
        with torch.no_grad():
            sensor_in = sensor_tensor.to(self.device)
            context_in = context_tensor.to(self.device)
            pred = model(sensor_in, context_in)
            loss = torch.mean((pred - sensor_in)**2).item()
        return loss
