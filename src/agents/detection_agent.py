import pandas as pd
import numpy as np

class DetectionAgent:
    """
    순수 원본 스트림을 받아서 감지만 수행하는 에이전트.
    이상치(Anomaly)는 "yellow", 오류(Fault)는 "red"로 분류하여 신호를 보냅니다.
    """
    def __init__(self):
        # 17개의 핵심 센서
        self.features = [
            'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
            'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
            'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
            'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
            'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
            'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
        ]
        # 실시간 Z-score 기반 감지를 위한 상태 저장 (간이 이동 평균)
        self.history = pd.DataFrame(columns=self.features)
        self.max_history = 1000  # 최근 1000개 데이터 기준
        
    def detect(self, incoming_chunk: pd.DataFrame) -> str:
        """
        들어온 스트림 데이터 청크를 평가하여 상태(normal, yellow, red) 반환
        """
        # 결측치가 있으면 이전 상태를 유지 (원본 데이터 특성 고려)
        chunk = incoming_chunk[self.features].ffill().bfill()
        
        # 아직 히스토리가 부족하면 정상으로 간주하고 데이터 누적
        if len(self.history) < 100:
            self.history = pd.concat([self.history, chunk]).tail(self.max_history)
            return "normal"
            
        # 히스토리의 평균과 표준편차 계산
        mean = self.history.mean()
        std = self.history.std() + 1e-6 # 0으로 나누기 방지
        
        # 현재 청크의 평균적인 센서값
        current_vals = chunk.mean()
        
        # Z-Score 계산 (얼마나 평소와 다른가?)
        z_scores = np.abs((current_vals - mean) / std)
        max_z = z_scores.max() # 가장 크게 튀는 센서의 Z-score
        
        # 데이터를 업데이트
        self.history = pd.concat([self.history, chunk]).tail(self.max_history)
        
        # 룰 베이스: 이상치(Yellow)와 오류(Red) 구분
        if max_z > 5.0:
            return "red"     # 심각한 오류 (Fault)
        elif max_z > 3.0:
            return "yellow"  # 이상치 경고 (Anomaly)
        else:
            return "normal"  # 정상
