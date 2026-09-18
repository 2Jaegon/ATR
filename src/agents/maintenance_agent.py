from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from src.agents.graph_state import AgentState

class MaintenanceAgent:
    def __init__(self, model_name="qwen2.5:1.5b"): # 경량 로컬 Qwen 모델 지정
        self.llm = ChatOllama(model=model_name, temperature=0) # 평가를 위해 창의성 0
        
    def evaluate_feedback(self, state: AgentState) -> dict:
        """
        엔지니어의 수리 보고서와 AI가 생성했던 Action Report를 비교하여
        정확도 및 가성 불량 여부를 평가하는 노드 함수
        """
        print("--- [Node: Maintenance Agent] Evaluating Engineer Feedback ---")
        action_report = state.get("action_report", "")
        feedback = state.get("engineer_feedback", "")
        
        system_prompt = (
            "당신은 공장 정비 결과 평가자입니다.\n"
            "AI가 예측한 'Action Report'와 현장 엔지니어가 작성한 '실제 수리 보고서'를 비교 분석하세요.\n"
            "결과는 두 가지를 포함해야 합니다:\n"
            "1. AI의 예측이 맞았는지 (일치 / 불일치)\n"
            "2. 가성 불량 여부 (센서 노이즈 등 실제 설비 고장이 아니었다면 True, 설비 고장이 맞았다면 False)\n"
            "출력 형식: [평가 요약] ... / [가성불량] True 또는 False"
        )
        
        user_prompt = f"[AI Action Report]\n{action_report}\n\n[실제 수리 보고서]\n{feedback}"
        
        try:
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            evaluation = response.content
            
            # 텍스트 파싱을 통한 가성 불량 Boolean 추출 (단순화된 예시)
            is_false_alarm = "True" in evaluation or "true" in evaluation
            
        except Exception as e:
            evaluation = f"LLM 연결 실패: {e}"
            is_false_alarm = False
            
        return {"evaluation_result": evaluation, "is_false_alarm": is_false_alarm, "next_step": "update_db"}
