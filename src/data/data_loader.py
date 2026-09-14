import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

class StreamingDataLoader:
    def __init__(self, data_file, fault_file=None, chunksize=10000):
        self.data_file = data_file
        self.chunksize = chunksize
        self.scaler = StandardScaler()
        self.is_scaler_fitted = False
        
        # 원본 파일 청크 단위 리더 생성
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Raw data file not found: {data_file}")
            
        self.reader = pd.read_csv(data_file, chunksize=self.chunksize)
        
        # 고장 정답지 로드 (평가/채점용)
        self.faults = pd.read_csv(fault_file) if fault_file and os.path.exists(fault_file) else pd.DataFrame()
        
        # 센서 데이터 피처 (공정 정보 제외)
        self.features = [
            'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
            'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
            'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
            'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
            'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
            'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
        ]

    def process_chunk(self, chunk):
        # 1. 시간 순서 보장 (원본 자체가 시간순이라 가정)
        chunk = chunk.sort_values('time').reset_index(drop=True)
        
        # 2. 결측치 처리 (원본에 있는 NaN을 이전 값으로 채움)
        chunk[self.features] = chunk[self.features].ffill().bfill()
        
        # 3. 라벨링 초기화
        chunk['label'] = 0
        chunk['fault_name'] = "None"
        
        # 4. 고장 라벨 부여 (실시간 매핑)
        if not self.faults.empty:
            for _, fault in self.faults.iterrows():
                f_time = fault['time']
                f_name = fault['fault_name']
                
                # 청크 내에 고장 시간이 포함되는지 확인
                if chunk['time'].iloc[0] <= f_time <= chunk['time'].iloc[-1] + 1000: # 여유 마진
                    target_idx = chunk[chunk['time'] <= f_time].index
                    if len(target_idx) > 0:
                        anomaly_indices = target_idx[-100:] # 직전 100행
                        chunk.loc[anomaly_indices, 'label'] = 1
                        chunk.loc[anomaly_indices, 'fault_name'] = f_name
        
        # 5. 스케일링 (첫 청크로 기준을 잡거나 지속적으로 업데이트)
        if not self.is_scaler_fitted:
            self.scaler.fit(chunk[self.features])
            self.is_scaler_fitted = True
            
        chunk[self.features] = self.scaler.transform(chunk[self.features])
        
        return chunk
        
    def __iter__(self):
        for chunk in self.reader:
            yield self.process_chunk(chunk)
