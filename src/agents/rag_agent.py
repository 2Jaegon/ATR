import pandas as pd
import numpy as np
import os
import oracledb
from dotenv import load_dotenv

class OracleRAGAgent:
    def __init__(self):
        load_dotenv()
        self.user = os.getenv("ORACLE_USER")
        self.password = os.getenv("ORACLE_PASSWORD")
        self.dsn = os.getenv("ORACLE_DSN")
        
    def get_connection(self):
        if not all([self.user, self.password, self.dsn]):
            raise ValueError("Oracle DB connection info is missing in .env")
        return oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)
        
    def search_similar_pattern(self, anomaly_data: pd.DataFrame):
        """
        Phase 3: Oracle DB(RDB)에 쿼리를 날려 가장 유사한 통계값을 가진 이력을 검색
        """
        target_temp = anomaly_data['temperature'].mean()
        target_vib = anomaly_data['vibration'].mean()
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # 유클리드 거리 기반 유사도 계산 쿼리 (온도와 진동의 차이 제곱합이 가장 작은 레코드 1개 추출)
            # Oracle 12c 이상 문법 (FETCH FIRST 1 ROWS ONLY)
            query = """
            SELECT id, pattern_desc, temp_mean, vib_mean, cause, action_taken, success_rate 
            FROM maintenance_history 
            ORDER BY POWER(temp_mean - :1, 2) + POWER(vib_mean - :2, 2) ASC
            FETCH FIRST 1 ROWS ONLY
            """
            
            cursor.execute(query, (target_temp, target_vib))
            row = cursor.fetchone()
            
            if row:
                best_match = {
                    "id": row[0],
                    "pattern_desc": row[1],
                    "temp_mean": row[2],
                    "vib_mean": row[3],
                    "cause": row[4],
                    "action": row[5],
                    "success_rate": row[6]
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
                "temp_mean": target_temp,
                "vib_mean": target_vib,
                "cause": "Unknown",
                "action": "Check DB Connection",
                "success_rate": 0.0
            }
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()
    
    def generate_action_report(self, anomaly_data: pd.DataFrame):
        # LangGraph 구조 이전의 구형 메서드 (호환성 유지)
        match = self.search_similar_pattern(anomaly_data)
        
        if "error" in match:
            return match["error"]
            
        report = f"""
        [Action Report]
        - 유사 과거 사례 ID: {match['id']}
        - 예상 원인: {match['cause']}
        - 권장 조치: {match['action']}
        - 조치 성공률: {match['success_rate']*100}%
        """
        return report

if __name__ == "__main__":
    # Test Oracle RAG
    mock_anomaly = pd.DataFrame({
        'temperature': [64.0, 65.5, 66.0],
        'vibration': [1.1, 1.2, 1.3]
    })
    
    agent = OracleRAGAgent()
    print(agent.generate_action_report(mock_anomaly))
