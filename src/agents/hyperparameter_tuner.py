import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from src.models.transformer_ae import TransformerAE
from src.models.gnn_ae import SpatialGNNAE
from src.models.lstm_ae import LSTMAE

class OptunaTuner:
    """
    딥러닝 3대장(Transformer, GNN, LSTM)의 최적 하이퍼파라미터를 
    Time-Series Cross Validation 기법으로 탐색하는 튜닝 에이전트.
    """
    def __init__(self, seq_len=5, n_trials=3):
        # n_trials: 너무 오래 걸리지 않도록 데모용으로는 3~5회만 탐색
        self.seq_len = seq_len
        self.n_trials = n_trials
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_windows(self, data: np.ndarray) -> torch.Tensor:
        windows = []
        for i in range(0, len(data) - self.seq_len + 1, self.seq_len):
            windows.append(data[i:i + self.seq_len])
        return torch.FloatTensor(np.array(windows))

    def _train_and_evaluate(self, model, X_train, X_val, lr, epochs=5):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # 모델 학습 (빠른 튜닝을 위해 epoch를 적게 설정)
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, X_train)
            loss.backward()
            optimizer.step()
            
        # 검증 평가
        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, X_val).item()
            
        return val_loss

    def optimize_transformer(self, normal_data: np.ndarray):
        print("\n[Tuner] Transformer-AE 하이퍼파라미터 탐색 시작...")
        
        def objective(trial):
            d_model = trial.suggest_categorical("d_model", [32, 64])
            nhead = trial.suggest_categorical("nhead", [2, 4])
            num_layers = trial.suggest_int("num_layers", 1, 2)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            
            tscv = TimeSeriesSplit(n_splits=3)
            val_losses = []
            
            for train_index, val_index in tscv.split(normal_data):
                X_tr = self._create_windows(normal_data[train_index]).to(self.device)
                X_va = self._create_windows(normal_data[val_index]).to(self.device)
                
                # 최소한의 데이터도 안 나오면 패스
                if len(X_tr) == 0 or len(X_va) == 0: continue
                
                model = TransformerAE(d_model=d_model, nhead=nhead, num_layers=num_layers).to(self.device)
                loss = self._train_and_evaluate(model, X_tr, X_va, lr)
                val_losses.append(loss)
                
            return np.mean(val_losses) if val_losses else 999.0

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        print(f"  -> Transformer 최적 파라미터: {study.best_params}")
        return study.best_params

    def optimize_gnn(self, normal_data: np.ndarray):
        print("\n[Tuner] Spatial GNN-AE 하이퍼파라미터 탐색 시작...")
        
        def objective(trial):
            hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            
            tscv = TimeSeriesSplit(n_splits=3)
            val_losses = []
            
            for train_index, val_index in tscv.split(normal_data):
                X_tr = self._create_windows(normal_data[train_index]).to(self.device)
                X_va = self._create_windows(normal_data[val_index]).to(self.device)
                
                if len(X_tr) == 0 or len(X_va) == 0: continue
                
                model = SpatialGNNAE(hidden_dim=hidden_dim).to(self.device)
                loss = self._train_and_evaluate(model, X_tr, X_va, lr)
                val_losses.append(loss)
                
            return np.mean(val_losses) if val_losses else 999.0

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        print(f"  -> GNN 최적 파라미터: {study.best_params}")
        return study.best_params

    def optimize_lstm(self, normal_data: np.ndarray):
        print("\n[Tuner] Seq2Seq LSTM-AE 하이퍼파라미터 탐색 시작...")
        
        def objective(trial):
            hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
            num_layers = trial.suggest_int("num_layers", 1, 2)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            
            tscv = TimeSeriesSplit(n_splits=3)
            val_losses = []
            
            for train_index, val_index in tscv.split(normal_data):
                X_tr = self._create_windows(normal_data[train_index]).to(self.device)
                X_va = self._create_windows(normal_data[val_index]).to(self.device)
                
                if len(X_tr) == 0 or len(X_va) == 0: continue
                
                model = LSTMAE(hidden_dim=hidden_dim, num_layers=num_layers).to(self.device)
                loss = self._train_and_evaluate(model, X_tr, X_va, lr)
                val_losses.append(loss)
                
            return np.mean(val_losses) if val_losses else 999.0

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.n_trials)
        print(f"  -> LSTM 최적 파라미터: {study.best_params}")
        return study.best_params
