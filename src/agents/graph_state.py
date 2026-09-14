from typing import TypedDict, List, Optional, Dict, Any

class AgentState(TypedDict):
    # 입력 데이터 (감지된 센서 이상치 기록)
    anomaly_data: Optional[Dict[str, Any]]
    
    # RAG 검색 결과
    retrieved_history: Optional[List[Dict[str, Any]]]
    
    # 생성된 Action Report (Main Agent 출력)
    action_report: Optional[str]
    
    # 엔지니어의 실제 수리 보고서 
    engineer_feedback: Optional[str]
    
    # Maintenance Agent의 분석 결과 (가성 불량 여부, 원인 일치 여부 등)
    evaluation_result: Optional[str]
    is_false_alarm: Optional[bool]
    
    # 다음 진행 방향 (라우팅용)
    next_step: Optional[str]
