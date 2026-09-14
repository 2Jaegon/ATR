import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class DetectionAgent:
    """
    순수 머신러닝(ML) 기반 감지 에이전트.
    초기화 시 원본 데이터와 정답지(faults)를 바탕으로 학습한 뒤,
    스트리밍되는 실시간 데이터의 상태(normal, yellow, red)를 빠르고 정확하게 판별합니다.
    (LLM을 사용하지 않는 순수 ML 영역)
    """
    def __init__(self):
        self.features = [
            'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
            'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
            'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
            'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
            'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
            'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
        ]
        self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, data_file: str, fault_file: str):
        print(f"[Detection Agent ML] 학습을 시작합니다... (이 과정은 최초 1회만 수행됩니다)")
        
        # 정답지 로드
        faults_df = pd.read_csv(fault_file)
        if faults_df.empty:
            raise ValueError("정답지(faults)가 비어있습니다.")
            
        first_fault_time = faults_df['time'].iloc[0]
        
        print(f"[Detection Agent ML] 원본 데이터({os.path.basename(data_file)}) 스캔 중...")
        # 전체를 다 읽으면 너무 오래 걸리므로, 앞부분(약 100만 줄)만 읽어서 학습 데이터를 추출합니다.
        # 현업에서는 하둡이나 스파크 등을 사용하지만, PoC를 위해 청크 리딩 사용
        normal_data = pd.DataFrame()
        fault_data = pd.DataFrame()
        
        chunk_iter = pd.read_csv(data_file, chunksize=50000)
        for chunk in chunk_iter:
            chunk = chunk.ffill().bfill()
            
            # 1. 정상 데이터 수집 (처음 5000개만)
            if len(normal_data) < 5000:
                normal_data = pd.concat([normal_data, chunk.head(5000)])
                
            # 2. 고장 데이터 수집 (정답지의 고장 시간 직전 데이터)
            if chunk['time'].iloc[0] <= first_fault_time <= chunk['time'].iloc[-1]:
                target_idx = chunk[chunk['time'] <= first_fault_time].index
                if len(target_idx) > 0:
                    # 고장 직전 100개의 데이터를 '위험(고장 전조)'으로 추출
                    fault_chunk = chunk.loc[target_idx[-100:]].copy()
                    fault_data = pd.concat([fault_data, fault_chunk])
                    break # 고장 데이터를 찾았으면 스캔 종료
                    
        # 학습용 데이터셋 구성 (Label: 0=Normal, 1=Fault/Anomaly)
        normal_data['label'] = 0
        fault_data['label'] = 1
        
        train_df = pd.concat([normal_data, fault_data]).reset_index(drop=True)
        X = train_df[self.features]
        y = train_df['label']
        
        print(f"[Detection Agent ML] 정답지를 바탕으로 Random Forest 모델 학습 중... (정상: {len(normal_data)}개, 고장 전조: {len(fault_data)}개)")
        
        # 모델 안정성을 위해 스케일링 적용 (내부 ML 모델용)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self.is_trained = True
        print("[Detection Agent ML] 머신러닝 두뇌 탑재 완료!\n")
        
    def detect(self, incoming_chunk: pd.DataFrame) -> str:
        """
        스트리밍으로 들어오는 센서 청크를 ML 모델이 평가하여 상태 반환
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")
            
        # 원본 데이터 그대로 받아서 내부적으로만 처리
        chunk = incoming_chunk[self.features].ffill().bfill()
        X_scaled = self.scaler.transform(chunk)
        
        # ML 모델 추론 (확률 반환)
        # 클래스 1(고장 전조)일 확률
        fault_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        max_prob = fault_probabilities.max()
        
        # ML 확률 기반 분류 (사용자 요구사항 반영)
        if max_prob > 0.8:
            return "red"     # 80% 이상 확신: 오류 (Fault)
        elif max_prob > 0.4:
            return "yellow"  # 40% 이상 확신: 이상치 경고 (Anomaly)
        else:
            return "normal"  # 정상
