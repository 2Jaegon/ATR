import pandas as pd
import numpy as np
import os
import sqlite3

class OracleRAGAgent:
    def __init__(self):
        self.db_path = "factory_logs.db"
        
        # 17개의 핵심 센서 리스트 (PHM 2018 기반)
        self.features = [
            'IONGAUGEPRESSURE', 'ETCHBEAMVOLTAGE', 'ETCHBEAMCURRENT', 
            'ETCHSUPPRESSORVOLTAGE', 'ETCHSUPPRESSORCURRENT', 'FLOWCOOLFLOWRATE', 
            'FLOWCOOLPRESSURE', 'ETCHGASCHANNEL1READBACK', 'ETCHPBNGASREADBACK', 
            'FIXTURETILTANGLE', 'ROTATIONSPEED', 'ACTUALROTATIONANGLE', 
            'FIXTURESHUTTERPOSITION', 'ETCHSOURCEUSAGE', 'ETCHAUXSOURCETIMER', 
            'ETCHAUX2SOURCETIMER', 'ACTUALSTEPDURATION'
        ]
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def search_similar_pattern(self, anomaly_data: pd.DataFrame):
        """
        Phase 5: 17차원 센서 데이터에 대한 유클리드 거리 기반 가장 유사한 과거 정비 이력 검색
        """
        # 현재 이상 데이터 윈도우의 17개 센서 평균값 도출
        target_values = []
        for feat in self.features:
            val = float(anomaly_data[feat].mean()) if feat in anomaly_data.columns else 0.0
            target_values.append(val)
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 동적으로 다차원(17차원) 거리 쿼리 생성 (SQLite 지원 형태)
            distance_terms = []
            for i, feat in enumerate(self.features):
                distance_terms.append(f"(({feat} - ?) * ({feat} - ?))")
            
            distance_calc = " + ".join(distance_terms)
            
            query = f"""
            SELECT id, pattern_desc, cause, action_taken, success_rate 
            FROM maintenance_history 
            ORDER BY {distance_calc} ASC
            LIMIT 1
            """
            
            # 각각의 ? 에 맵핑하기 위해 값을 2번씩 반복
            query_params = []
            for val in target_values:
                query_params.extend([val, val])
                
            cursor.execute(query, tuple(query_params))
            row = cursor.fetchone()
            
            if row:
                best_match = {
                    "id": row[0],
                    "pattern_desc": row[1],
                    "cause": row[2],
                    "action": row[3],
                    "success_rate": row[4]
                }
                return best_match
            else:
                return {"error": "No maintenance history found in DB."}
                
        except Exception as e:
            print(f"[RAG Agent DB Error] {e}")
            # DB 연결 실패시 Fallback Mock Data 반환
            return {
                "id": -1,
                "pattern_desc": "Fallback (DB Connection Failed)",
                "cause": "Unknown Database Error",
                "action": "Check DB Connection",
                "success_rate": 0.0
            }
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def generate_action_report(self, anomaly_data: pd.DataFrame):
        match = self.search_similar_pattern(anomaly_data)
        
        if "error" in match:
            return match["error"]
            
        report = f"""
        [Action Report]
        - 유사 과거 사례 ID: {match['id']}
        - 감지된 패턴: {match['pattern_desc']}
        - 예상 원인: {match['cause']}
        - 권장 조치: {match['action']}
        - 조치 성공률: {match['success_rate']*100}%
        """
        return report

if __name__ == "__main__":
    # Test Oracle RAG with mock 17d dataframe
    mock_dict = {f: [np.random.rand()] for f in OracleRAGAgent().features}
    mock_anomaly = pd.DataFrame(mock_dict)
    
    agent = OracleRAGAgent()
    print(agent.generate_action_report(mock_anomaly))
