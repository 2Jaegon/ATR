import pandas as pd
import os

class RawDataLoader:
    """
    순수하게 원본 데이터를 읽어서 실시간 스트리밍(제너레이터) 역할만 수행합니다.
    (라벨링, 스케일링, 데이터 변형 일절 없음)
    """
    def __init__(self, data_file, chunksize=5):
        self.data_file = data_file
        self.chunksize = chunksize
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Raw data file not found: {data_file}")
            
        self.reader = pd.read_csv(data_file, chunksize=self.chunksize)

    def __iter__(self):
        for chunk in self.reader:
            yield chunk
