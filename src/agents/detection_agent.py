import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from src.models.transformer_ae import TransformerAE

class DetectionAgent:
    """
    SOTA 딥러닝(Transformer Autoencoder) 기반 감지 에이전트.
    초기화 시 정상 데이터를 학습하여 센서 간의 시간/공간적 인과관계를 파악(복원)하는 능력을 기릅니다.
    실시간 데이터 복원 오차(Reconstruction Error)를 계산하여 이상 상태를 판별합니다.
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
        
        self.model = TransformerAE(num_features=len(self.features)).to(self.device)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.yellow_threshold = 0.0
        self.red_threshold = 0.0
        
    def _create_windows(self, data: np.ndarray) -> torch.Tensor:
        """
        시계열 데이터를 시퀀스 길이(seq_len)만큼 잘라서 윈도우 생성
        shape: (num_windows, seq_len, features)
        """
        windows = []
        for i in range(0, len(data) - self.seq_len + 1, self.seq_len):
            windows.append(data[i:i + self.seq_len])
        return torch.FloatTensor(np.array(windows))
        
    def train(self, data_file: str, fault_file: str):
        print(f"[Detection Agent 딥러닝] 시계열 Transformer Autoencoder 학습 준비 중... (Device: {self.device})")
        
        faults_df = pd.read_csv(fault_file)
        first_fault_time = faults_df['time'].iloc[0]
        
        normal_data_list = []
        fault_data_list = []
        
        # 데이터 수집 (청크 리딩)
        chunk_iter = pd.read_csv(data_file, chunksize=50000)
        for chunk in chunk_iter:
            chunk = chunk.ffill().bfill()
            
            # 1. 정상 데이터 수집 (처음 20000개 정도만 학습에 사용)
            if len(normal_data_list) == 0:
                normal_data_list.append(chunk.head(20000))
                
            # 2. 고장 전조 데이터 수집 (임계치 설정용)
            if chunk['time'].iloc[0] <= first_fault_time <= chunk['time'].iloc[-1]:
                target_idx = chunk[chunk['time'] <= first_fault_time].index
                if len(target_idx) > 0:
                    fault_data_list.append(chunk.loc[target_idx[-100:]])
                    break
                    
        normal_df = pd.concat(normal_data_list)
        fault_df = pd.concat(fault_data_list)
        
        # 스케일링 (정상 데이터 기준으로 핏팅)
        normal_scaled = self.scaler.fit_transform(normal_df[self.features])
        fault_scaled = self.scaler.transform(fault_df[self.features])
        
        # 시퀀스 텐서로 변환
        X_train = self._create_windows(normal_scaled).to(self.device)
        X_fault = self._create_windows(fault_scaled).to(self.device)
        
        print(f"[Detection Agent 딥러닝] 정상 시퀀스 {len(X_train)}개, 고장 전조 시퀀스 {len(X_fault)}개 추출 완료")
        print("[Detection Agent 딥러닝] Transformer 모델 복원(Reconstruction) 학습 시작...")
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 학습 루프 (Epoch 20)
        self.model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = criterion(output, X_train)
            loss.backward()
            optimizer.step()
            
        print(f"[Detection Agent 딥러닝] 학습 완료! 최종 Normal MSE Loss: {loss.item():.4f}")
        
        # 임계치(Threshold) 설정
        self.model.eval()
        with torch.no_grad():
            # 정상 데이터의 오차 분포
            normal_pred = self.model(X_train)
            normal_loss = torch.mean((normal_pred - X_train)**2, dim=(1,2)).cpu().numpy()
            
            # 고장 전조 데이터의 오차 분포
            if len(X_fault) > 0:
                fault_pred = self.model(X_fault)
                fault_loss = torch.mean((fault_pred - X_fault)**2, dim=(1,2)).cpu().numpy()
                self.red_threshold = np.min(fault_loss) * 0.8 # 고장 데이터 오차의 80% 선을 빨간점 기준으로
            else:
                self.red_threshold = np.max(normal_loss) * 5.0
                
            self.yellow_threshold = np.percentile(normal_loss, 99) # 정상 데이터 상위 1%를 노란점 기준으로
            
        print(f"[Detection Agent 딥러닝] 임계치 설정 완료 - Yellow(Anomaly): {self.yellow_threshold:.4f}, Red(Fault): {self.red_threshold:.4f}\n")
        self.is_trained = True
        
    def detect(self, incoming_chunk: pd.DataFrame) -> str:
        """
        스트리밍 청크(길이 5)를 모델에 통과시켜 복원 오차 계산
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다.")
            
        # 정확히 모델이 요구하는 시퀀스 길이(seq_len)인지 확인
        if len(incoming_chunk) != self.seq_len:
            return "normal"
            
        chunk = incoming_chunk[self.features].ffill().bfill()
        scaled_data = self.scaler.transform(chunk)
        
        # 차원 맞추기 (1, seq_len, features)
        x_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x_tensor)
            # MSE Loss 계산
            loss = torch.mean((pred - x_tensor)**2).item()
            
        if loss > self.red_threshold:
            return "red"
        elif loss > self.yellow_threshold:
            return "yellow"
        else:
            return "normal"
