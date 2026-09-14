import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

class DetectionAgent:
    def __init__(self, data_path="data/training_core.csv"):
        self.data_path = data_path
        self.model = None
        
        # 17개의 핵심 센서 리스트 (PHM 2018 기반)
        self.features = [
            'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
            'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
            'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
            'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
            'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
            'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
        ]
        
    def train(self):
        """
        Phase 5: 17차원 다변량 데이터 기반 Isolation Forest 학습
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Core training data not found at {self.data_path}. Please run preprocess_phm.py first.")
            
        print("[Detection Agent] Loading core dataset for training...")
        df = pd.read_csv(self.data_path)
        
        # 정상 데이터(label == 0)만 추출하여 정상 패턴 학습
        normal_data = df[df['label'] == 0][self.features]
        
        print(f"[Detection Agent] Training Isolation Forest on {len(normal_data)} normal samples with {len(self.features)} features...")
        # 오염도(contamination)는 아주 낮게 설정하여 극단적인 이상치만 잡아내도록 함
        self.model = IsolationForest(contamination=0.01, random_state=42)
        self.model.fit(normal_data)
        print("[Detection Agent] Training Complete.")
        
    def detect(self, incoming_data: pd.DataFrame) -> bool:
        """
        실시간으로 들어온 센서 데이터 프레임에서 이상이 있는지 판별
        """
        if self.model is None:
            self.train()
            
        # 모델 추론 (-1 이면 이상, 1 이면 정상)
        X = incoming_data[self.features]
        predictions = self.model.predict(X)
        
        # 단 하나의 시점이라도 이상치(-1)로 판별되면 True 반환
        return -1 in predictions
