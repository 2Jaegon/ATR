import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from src.models.transformer_ae import TransformerAE
from src.models.gnn_ae import SpatialGNNAE

class DetectionAgent:
    """
    SOTA 딥러닝 앙상블 (GNN + Transformer + ML) 하이브리드 감지 에이전트.
    """
    def __init__(self, seq_len=5):
        self.features = [
            'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
            'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
            'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
            'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
            'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
            'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
        ]
        self.seq_len = seq_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 3개의 SOTA 두뇌 초기화
        self.model_tr = TransformerAE(num_features=len(self.features)).to(self.device)
        self.model_gnn = SpatialGNNAE(num_features=len(self.features)).to(self.device)
        self.model_rf = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # 최적 앙상블 가중치 (초기값 균등)
        self.weights = {'TR': 0.33, 'GNN': 0.33, 'RF': 0.34}
        self.yellow_threshold = 0.0
        self.red_threshold = 0.0
        
    def _create_windows(self, data: np.ndarray) -> torch.Tensor:
        windows = []
        for i in range(0, len(data) - self.seq_len + 1, self.seq_len):
            windows.append(data[i:i + self.seq_len])
        return torch.FloatTensor(np.array(windows))
        
    def _train_pytorch_model(self, model, X_train, name, epochs=15):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        print(f"  [{name}] 학습 진행 중...")
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, X_train)
            loss.backward()
            optimizer.step()
        return model

    def _get_dl_scores(self, model, X_tensor):
        model.eval()
        with torch.no_grad():
            pred = model(X_tensor)
            loss = torch.mean((pred - X_tensor)**2, dim=(1,2)).cpu().numpy()
        return loss

    def train(self, data_file: str, fault_file: str):
        print(f"\n[Detection Agent 하이브리드 앙상블] 학습 및 최적화 시작! (Device: {self.device})")
        
        faults_df = pd.read_csv(fault_file)
        first_fault_time = faults_df['time'].iloc[0]
        
        normal_data_list = []
        fault_data_list = []
        
        chunk_iter = pd.read_csv(data_file, chunksize=50000)
        for chunk in chunk_iter:
            chunk = chunk.ffill().bfill()
            if len(normal_data_list) == 0:
                normal_data_list.append(chunk.head(10000)) # 정상 데이터 1만개 추출
                
            if chunk['time'].iloc[0] <= first_fault_time <= chunk['time'].iloc[-1]:
                target_idx = chunk[chunk['time'] <= first_fault_time].index
                if len(target_idx) > 0:
                    fault_data_list.append(chunk.loc[target_idx[-100:]]) # 고장 데이터 100개 추출
                    break
                    
        normal_df = pd.concat(normal_data_list)
        fault_df = pd.concat(fault_data_list)
        
        # Validation 분리 (가중치 튜닝용)
        val_normal = normal_df.iloc[-2000:] 
        train_normal = normal_df.iloc[:-2000]
        
        # 1. 스케일링
        self.scaler.fit(train_normal[self.features])
        X_train_norm = self.scaler.transform(train_normal[self.features])
        X_val_norm = self.scaler.transform(val_normal[self.features])
        X_fault = self.scaler.transform(fault_df[self.features])
        
        # 2. 개별 모델 학습 (Phase 1)
        print("[Phase 1] 3개의 독립된 뇌(Transformer, GNN, Random Forest) 개별 학습")
        
        # 딥러닝 시퀀스 변환
        tensor_train_norm = self._create_windows(X_train_norm).to(self.device)
        tensor_val_norm = self._create_windows(X_val_norm).to(self.device)
        tensor_fault = self._create_windows(X_fault).to(self.device)
        
        # TR & GNN 학습 (정상 데이터만)
        self.model_tr = self._train_pytorch_model(self.model_tr, tensor_train_norm, "Transformer")
        self.model_gnn = self._train_pytorch_model(self.model_gnn, tensor_train_norm, "GNN")
        
        # RF 학습 (정상+고장 지도학습)
        print("  [Random Forest] 통계 기반 앙상블 학습 중...")
        rf_train_x = np.vstack([X_train_norm, X_fault])
        rf_train_y = np.array([0]*len(X_train_norm) + [1]*len(X_fault))
        self.model_rf.fit(rf_train_x, rf_train_y)
        
        # 3. 메타 러닝: 가중치 최적화 (Phase 2)
        print("\n[Phase 2] 정답률 기반 앙상블 가중치(Soft Voting Weights) 메타 최적화 (Test)")
        
        # 각 모델의 예측 점수(0~1 스케일링) 추출
        # 딥러닝: Loss (오차)
        tr_val_loss = self._get_dl_scores(self.model_tr, tensor_val_norm)
        tr_fault_loss = self._get_dl_scores(self.model_tr, tensor_fault)
        
        gnn_val_loss = self._get_dl_scores(self.model_gnn, tensor_val_norm)
        gnn_fault_loss = self._get_dl_scores(self.model_gnn, tensor_fault)
        
        # RF: Probability (확률)
        rf_val_prob = self.model_rf.predict_proba(X_val_norm)[:, 1]
        rf_fault_prob = self.model_rf.predict_proba(X_fault)[:, 1]
        
        # 스케일 정규화 (최대치 대비 비율)
        max_tr = np.max(np.concatenate([tr_val_loss, tr_fault_loss])) + 1e-6
        max_gnn = np.max(np.concatenate([gnn_val_loss, gnn_fault_loss])) + 1e-6
        
        tr_val_norm_scores = tr_val_loss / max_tr
        tr_fault_norm_scores = tr_fault_loss / max_tr
        
        gnn_val_norm_scores = gnn_val_loss / max_gnn
        gnn_fault_norm_scores = gnn_fault_loss / max_gnn
        
        # RF 점수는 시퀀스에 맞게 평균냄
        rf_val_seq = [np.mean(rf_val_prob[i:i+self.seq_len]) for i in range(0, len(rf_val_prob)-self.seq_len+1, self.seq_len)]
        rf_fault_seq = [np.mean(rf_fault_prob[i:i+self.seq_len]) for i in range(0, len(rf_fault_prob)-self.seq_len+1, self.seq_len)]
        
        # 그리드 서치를 통한 최적 가중치 탐색 (정상 데이터의 점수는 최소로, 고장 데이터 점수는 최대로)
        best_w = self.weights
        best_margin = -999.0
        
        weights_grid = np.arange(0.0, 1.1, 0.1)
        for w1 in weights_grid:
            for w2 in weights_grid:
                w3 = 1.0 - w1 - w2
                if w3 < -0.01: continue
                
                # 가중합 점수 계산
                ens_val = w1 * tr_val_norm_scores + w2 * gnn_val_norm_scores + w3 * np.array(rf_val_seq)
                ens_fault = w1 * tr_fault_norm_scores + w2 * gnn_fault_norm_scores + w3 * np.array(rf_fault_seq)
                
                # 고장 점수의 평균과 정상 점수의 최댓값 간의 격차(Margin)가 클수록 우수한 가중치 조합
                margin = np.mean(ens_fault) - np.max(ens_val)
                if margin > best_margin:
                    best_margin = margin
                    best_w = {'TR': w1, 'GNN': w2, 'RF': w3}
                    
        self.weights = best_w
        print(f"  --> [최적화 완료] 도출된 가중치: Transformer({self.weights['TR']:.2f}) / GNN({self.weights['GNN']:.2f}) / Random Forest({self.weights['RF']:.2f})")
        
        # 최종 임계치 설정
        self.max_tr = max_tr
        self.max_gnn = max_gnn
        
        # 고장 데이터 점수의 70% 선을 Red로 설정
        final_fault_scores = self.weights['TR'] * tr_fault_norm_scores + self.weights['GNN'] * gnn_fault_norm_scores + self.weights['RF'] * np.array(rf_fault_seq)
        final_val_scores = self.weights['TR'] * tr_val_norm_scores + self.weights['GNN'] * gnn_val_norm_scores + self.weights['RF'] * np.array(rf_val_seq)
        
        self.red_threshold = np.min(final_fault_scores) * 0.7
        self.yellow_threshold = np.percentile(final_val_scores, 99) # 정상의 상위 1%
        
        self.is_trained = True
        print("[Detection Agent] 하이브리드 감시 시스템 준비 완료!\n")
        
    def detect(self, incoming_chunk: pd.DataFrame) -> str:
        """
        스트리밍 데이터에 대해 3개 모델이 앙상블 추론 후 최종 결과(노란점/빨간점) 반환
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        if len(incoming_chunk) != self.seq_len:
            return "normal"
            
        chunk = incoming_chunk[self.features].ffill().bfill()
        scaled_data = self.scaler.transform(chunk)
        
        x_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
        
        # 개별 모델 점수 계산
        tr_loss = self._get_dl_scores(self.model_tr, x_tensor)[0] / self.max_tr
        gnn_loss = self._get_dl_scores(self.model_gnn, x_tensor)[0] / self.max_gnn
        rf_prob = np.mean(self.model_rf.predict_proba(scaled_data)[:, 1])
        
        # 하이브리드 소프트 보팅 (가중합)
        final_score = self.weights['TR'] * tr_loss + self.weights['GNN'] * gnn_loss + self.weights['RF'] * rf_prob
        
        if final_score > self.red_threshold:
            return "red"
        elif final_score > self.yellow_threshold:
            return "yellow"
        else:
            return "normal"
