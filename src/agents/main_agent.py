from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.graph_state import AgentState

class MainAgent:
    def __init__(self, model_name="qwen2.5:1.5b"): # 경량 로컬 Qwen 모델 지정 (VRAM/RAM < 1.5GB)
        # Ollama 서버(localhost:11434)와 통신하는 랭체인 객체
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        
    def generate_action_report(self, state: AgentState) -> dict:
        """
        RAG에서 검색된 과거 이력을 바탕으로 Action Report를 생성하는 노드 함수
        """
        print("--- [Node: Main Agent] Generating Action Report ---")
        anomaly = state.get("anomaly_data", {})
        history = state.get("retrieved_history", [])
        
        system_prompt = (
            "당신은 공장 설비의 센서 데이터를 분석하는 수석 엔지니어입니다.\n"
            "주어진 과거 유사 정비 이력을 참고하여 현장 엔지니어에게 명확한 'Action Report'를 작성하세요.\n"
            "형식은 [예상 원인], [권장 조치 내역], [신뢰도]를 포함해야 합니다."
        )
        
        user_prompt = f"현재 감지된 이상 센서 평균값: 온도 {anomaly.get('temp_mean')}, 진동 {anomaly.get('vib_mean')}\n"
        user_prompt += f"검색된 과거 유사 이력: {history}\n\n"
        user_prompt += "이 정보를 바탕으로 간결한 Action Report를 작성해 주세요."
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            action_report = response.content
        except Exception as e:
            action_report = f"LLM 연결 실패 (Ollama 서버를 확인하세요): {e}\n\n[Fallback Report] 예상 원인: 알 수 없음, 조치: 점검 요망"

        # State 업데이트를 위해 딕셔너리 반환
        return {"action_report": action_report, "next_step": "wait_for_engineer"}
