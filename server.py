import asyncio
import json
import random
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_index():
    return FileResponse("static/index.html")

print("[Server] AI 대시보드 서버 초기화 중... (Torch 환경 제약으로 Mock Streaming 모드 가동)")

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    await websocket.send_json({
        "type": "init",
        "weights": {"TR": 0.40, "GNN": 0.35, "LSTM": 0.25},
        "yellow_threshold": 0.6,
        "red_threshold": 0.8
    })
    
    print("[Server] 클라이언트 연결! 스트리밍 시작...")
    try:
        base_pressure = 50.0
        
        while True:
            # 약간의 노이즈 추가
            base_pressure += random.uniform(-0.5, 0.5)
            
            tr_score = random.uniform(0.1, 0.4)
            gnn_score = random.uniform(0.1, 0.4)
            lstm_score = random.uniform(0.1, 0.4)
            
            # 이벤트 시뮬레이션
            chance = random.random()
            if chance < 0.03: # 3% 확률로 고장(Red)
                tr_score += 0.6; gnn_score += 0.5; lstm_score += 0.6
                base_pressure += random.uniform(10.0, 15.0) # 압력 폭증
            elif chance < 0.1: # 10% 확률로 이상(Yellow)
                tr_score += 0.3; gnn_score += 0.3; lstm_score += 0.4
                base_pressure -= random.uniform(5.0, 8.0) # 압력 감소
                
            final_score = 0.4*tr_score + 0.35*gnn_score + 0.25*lstm_score
            
            status = "normal"
            if final_score > 0.8:
                status = "red"
            elif final_score > 0.6:
                status = "yellow"
                
            await websocket.send_json({
                "type": "telemetry",
                "status": status,
                "final_score": final_score,
                "scores": {"TR": tr_score, "GNN": gnn_score, "LSTM": lstm_score},
                "sensor_value": float(base_pressure)
            })
            
            if status == 'yellow':
                await websocket.send_json({
                    "type": "report",
                    "status": "yellow",
                    "message": "[경고] Flowcool Pressure 비정상 하강 감지. 과거 이력 분석: 밸브 미세 누수 의심."
                })
            elif status == 'red':
                await websocket.send_json({
                    "type": "report",
                    "status": "red",
                    "message": "[CRITICAL] 압력 한계선 돌파! 즉각적인 장비 가동 중단(Interlock) 및 펌프 교체 요망!"
                })
            
            await asyncio.sleep(0.5) # 초당 2프레임으로 전송
            
    except WebSocketDisconnect:
        print("[Server] 클라이언트 연결 종료")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
