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
        
    def train(self, train_df: pd.DataFrame):
        """
        Phase 5: 17차원 다변량 데이터 기반 Isolation Forest 학습
        (Streaming DataLoader에서 생성된 첫 번째 Chunk 등 연속된 정상 데이터를 받아서 학습)
        """
        print("[Detection Agent] Training Isolation Forest on normal samples with 17 features...")
        
        # 정상 데이터(label == 0)만 추출하여 정상 패턴 학습
        normal_data = train_df[train_df['label'] == 0][self.features]
        
        # 오염도(contamination)는 아주 낮게 설정하여 극단적인 이상치만 잡아내도록 함
        self.model = IsolationForest(contamination=0.01, random_state=42)
        self.model.fit(normal_data)
        print("[Detection Agent] Training Complete.")
        
    def detect(self, incoming_data: pd.DataFrame) -> bool:
        """
        실시간으로 들어온 센서 데이터 프레임에서 이상이 있는지 판별
        """
        if self.model is None:
            raise ValueError("Model is not trained yet. Call train() first.")
            
        # 모델 추론 (-1 이면 이상, 1 이면 정상)
        X = incoming_data[self.features]
        predictions = self.model.predict(X)
        
        # 단 하나의 시점이라도 이상치(-1)로 판별되면 True 반환
        return -1 in predictions
