import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler

from src.models.transformer_ae import TransformerAE
from src.models.gnn_ae import SpatialGNNAE
from src.models.lstm_ae import LSTMAE
from src.agents.hyperparameter_tuner import OptunaTuner

class DetectionAgent:
    """
    The Ultimate DL Trio 앙상블 (Transformer + GNN + LSTM) 
    + Optuna 하이퍼파라미터 자동 튜닝 (Method B) 탑재
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
        
        # 임시 초기화 (나중에 Optuna 튜닝 결과로 재조립됨)
        self.model_tr = None
        self.model_gnn = None
        self.model_lstm = None
        
        # 튜닝된 최적의 학습률(LR)을 저장
        self.best_lrs = {'TR': 0.001, 'GNN': 0.001, 'LSTM': 0.001}
        
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.weights = {'TR': 0.33, 'GNN': 0.33, 'LSTM': 0.34}
        self.yellow_threshold = 0.0
        self.red_threshold = 0.0
        
    def _create_windows(self, data: np.ndarray) -> torch.Tensor:
        windows = []
        for i in range(0, len(data) - self.seq_len + 1, self.seq_len):
            windows.append(data[i:i + self.seq_len])
        return torch.FloatTensor(np.array(windows))
        
    def _train_pytorch_model(self, model, X_train, name, lr, epochs=15):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        print(f"  [{name}] 최적 구조로 정식 학습 진행 중 (LR: {lr:.5f})...")
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
        print(f"\n[Detection Agent HPO] 모델 구조 최적화 및 앙상블 학습 시작! (Device: {self.device})")
        
        faults_df = pd.read_csv(fault_file)
        first_fault_time = faults_df['time'].iloc[0]
        
        normal_data_list = []
        fault_data_list = []
        
        chunk_iter = pd.read_csv(data_file, chunksize=50000)
        for chunk in chunk_iter:
            chunk = chunk.ffill().bfill()
            if len(normal_data_list) == 0:
                normal_data_list.append(chunk.head(10000)) 
                
            if chunk['time'].iloc[0] <= first_fault_time <= chunk['time'].iloc[-1]:
                target_idx = chunk[chunk['time'] <= first_fault_time].index
                if len(target_idx) > 0:
                    fault_data_list.append(chunk.loc[target_idx[-100:]])
                    break
                    
        normal_df = pd.concat(normal_data_list)
        fault_df = pd.concat(fault_data_list)
        
        val_normal = normal_df.iloc[-2000:] 
        train_normal = normal_df.iloc[:-2000]
        
        self.scaler.fit(train_normal[self.features])
        X_train_norm = self.scaler.transform(train_normal[self.features])
        X_val_norm = self.scaler.transform(val_normal[self.features])
        X_fault = self.scaler.transform(fault_df[self.features])
        
        # ---------------------------------------------------------
        # [NEW] Phase 0: Optuna 기반 하이퍼파라미터 최적화 (HPO)
        # ---------------------------------------------------------
        print("\n[Phase 0] Optuna 프레임워크 기반 하이퍼파라미터 자동 탐색 (Time-Series K-Fold)")
        tuner = OptunaTuner(seq_len=self.seq_len, n_trials=3) # 데모용으로 3회 탐색
        
        tr_params = tuner.optimize_transformer(X_train_norm)
        gnn_params = tuner.optimize_gnn(X_train_norm)
        lstm_params = tuner.optimize_lstm(X_train_norm)
        
        # 탐색된 파라미터로 모델 재조립
        self.model_tr = TransformerAE(
            d_model=tr_params['d_model'], nhead=tr_params['nhead'], num_layers=tr_params['num_layers']
        ).to(self.device)
        self.best_lrs['TR'] = tr_params['lr']
        
        self.model_gnn = SpatialGNNAE(
            hidden_dim=gnn_params['hidden_dim']
        ).to(self.device)
        self.best_lrs['GNN'] = gnn_params['lr']
        
        self.model_lstm = LSTMAE(
            hidden_dim=lstm_params['hidden_dim'], num_layers=lstm_params['num_layers']
        ).to(self.device)
        self.best_lrs['LSTM'] = lstm_params['lr']
        
        # ---------------------------------------------------------
        # Phase 1: 개별 모델 정식 학습 (재조립된 최적의 모델로)
        # ---------------------------------------------------------
        print("\n[Phase 1] 3개의 딥러닝 두뇌 정식 학습 (최적 파라미터 적용)")
        tensor_train_norm = self._create_windows(X_train_norm).to(self.device)
        tensor_val_norm = self._create_windows(X_val_norm).to(self.device)
        tensor_fault = self._create_windows(X_fault).to(self.device)
        
        self.model_tr = self._train_pytorch_model(self.model_tr, tensor_train_norm, "Transformer AE", self.best_lrs['TR'])
        self.model_gnn = self._train_pytorch_model(self.model_gnn, tensor_train_norm, "Spatial GNN AE", self.best_lrs['GNN'])
        self.model_lstm = self._train_pytorch_model(self.model_lstm, tensor_train_norm, "LSTM AE", self.best_lrs['LSTM'])
        
        # ---------------------------------------------------------
        # Phase 2: 메타 러닝 (앙상블 가중치 최적화)
        # ---------------------------------------------------------
        print("\n[Phase 2] 정답률 기반 앙상블 가중치(Soft Voting Weights) 메타 최적화")
        
        tr_val_loss = self._get_dl_scores(self.model_tr, tensor_val_norm)
        tr_fault_loss = self._get_dl_scores(self.model_tr, tensor_fault)
        
        gnn_val_loss = self._get_dl_scores(self.model_gnn, tensor_val_norm)
        gnn_fault_loss = self._get_dl_scores(self.model_gnn, tensor_fault)
        
        lstm_val_loss = self._get_dl_scores(self.model_lstm, tensor_val_norm)
        lstm_fault_loss = self._get_dl_scores(self.model_lstm, tensor_fault)
        
        max_tr = np.max(np.concatenate([tr_val_loss, tr_fault_loss])) + 1e-6
        max_gnn = np.max(np.concatenate([gnn_val_loss, gnn_fault_loss])) + 1e-6
        max_lstm = np.max(np.concatenate([lstm_val_loss, lstm_fault_loss])) + 1e-6
        
        tr_val_norm = tr_val_loss / max_tr
        tr_fault_norm = tr_fault_loss / max_tr
        
        gnn_val_norm = gnn_val_loss / max_gnn
        gnn_fault_norm = gnn_fault_loss / max_gnn
        
        lstm_val_norm = lstm_val_loss / max_lstm
        lstm_fault_norm = lstm_fault_loss / max_lstm
        
        best_w = self.weights
        best_margin = -999.0
        
        weights_grid = np.arange(0.0, 1.1, 0.1)
        for w1 in weights_grid:
            for w2 in weights_grid:
                w3 = 1.0 - w1 - w2
                if w3 < -0.01: continue
                
                ens_val = w1 * tr_val_norm + w2 * gnn_val_norm + w3 * lstm_val_norm
                ens_fault = w1 * tr_fault_norm + w2 * gnn_fault_norm + w3 * lstm_fault_norm
                
                margin = np.mean(ens_fault) - np.max(ens_val)
                if margin > best_margin:
                    best_margin = margin
                    best_w = {'TR': w1, 'GNN': w2, 'LSTM': w3}
                    
        self.weights = best_w
        print(f"  --> [가중치 최적화 완료] Transformer({self.weights['TR']:.2f}) / GNN({self.weights['GNN']:.2f}) / LSTM({self.weights['LSTM']:.2f})")
        
        self.max_tr = max_tr
        self.max_gnn = max_gnn
        self.max_lstm = max_lstm
        
        final_fault_scores = self.weights['TR'] * tr_fault_norm + self.weights['GNN'] * gnn_fault_norm + self.weights['LSTM'] * lstm_fault_norm
        final_val_scores = self.weights['TR'] * tr_val_norm + self.weights['GNN'] * gnn_val_norm + self.weights['LSTM'] * lstm_val_norm
        
        self.red_threshold = np.min(final_fault_scores) * 0.7
        self.yellow_threshold = np.percentile(final_val_scores, 99) 
        
        self.is_trained = True
        print("[Detection Agent] Optuna 튜닝이 접목된 궁극의 하이브리드 감시 시스템 준비 완료!\n")
        
    def detect(self, incoming_chunk: pd.DataFrame) -> str:
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        if len(incoming_chunk) != self.seq_len:
            return "normal"
            
        chunk = incoming_chunk[self.features].ffill().bfill()
        scaled_data = self.scaler.transform(chunk)
        
        x_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
        
        tr_loss = self._get_dl_scores(self.model_tr, x_tensor)[0] / self.max_tr
        gnn_loss = self._get_dl_scores(self.model_gnn, x_tensor)[0] / self.max_gnn
        lstm_loss = self._get_dl_scores(self.model_lstm, x_tensor)[0] / self.max_lstm
        
        final_score = self.weights['TR'] * tr_loss + self.weights['GNN'] * gnn_loss + self.weights['LSTM'] * lstm_loss
        
        if final_score > self.red_threshold:
            return "red"
        elif final_score > self.yellow_threshold:
            return "yellow"
        else:
            return "normal"
