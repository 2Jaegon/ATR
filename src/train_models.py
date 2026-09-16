"""
State-Aware 하이브리드 딥러닝 앙상블 학습 스크립트
===================================================
- Phase 1: RobustScaler + 문맥 인코더 피팅 (정상 데이터만)
- Phase 2: Conditional 모델 3종 스트리밍 학습 (OOM 방지)
- Phase 3: 정상 데이터 Anomaly Score 분포 수집 → POT 동적 임계값 산출
"""
import os
import glob
import numpy as np
import joblib
from tqdm import tqdm
import time
import ctypes

def prevent_sleep():
    try:
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[System] 절전 모드 방지 활성화")
    except Exception as e:
        print(f"[Warning] 절전 모드 방지 실패: {e}")

def allow_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("[System] 절전 모드 방지 해제")
    except Exception as e:
        pass

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from src.models.transformer_ae import ConditionalTransformerAE
    from src.models.gnn_ae import ConditionalSTGNNAE
    from src.models.lstm_ae import ConditionalLSTMAE
    TORCH_AVAILABLE = True
except OSError as e:
    print(f"[Warning] PyTorch DLL Error ({e}). Switching to Mock mode.")
    TORCH_AVAILABLE = False

from src.data.preprocessing import (
    PreprocessingPipeline, StreamingWindowDataset,
    SENSOR_FEATURES
)

# ─── 설정 ──────────────────────────────────────────────────
DATA_DIR = "data/phm_data_challenge_2018/train_preprocessing"
WEIGHTS_DIR = "src/models/weights"
SEQ_LEN = 5
BATCH_SIZE = 64
EPOCHS = 3
CHUNKSIZE = 100000

os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ─── 학습 함수 ────────────────────────────────────────────
def train_model(model, train_loader, optimizer, criterion, model_name, device, epochs=EPOCHS):
    """Conditional 모델 학습 루프 (sensor, context 쌍)"""
    save_path = os.path.join(WEIGHTS_DIR, f"{model_name.lower()}.pt")
    
    if os.path.exists(save_path):
        try:
            model.load_state_dict(torch.load(save_path, map_location=device))
            print(f"\n[{model_name}] 기존 가중치 로드 성공. (이어서 학습 진행)")
        except Exception as e:
            print(f"\n[{model_name}] 기존 가중치 로드 실패: {e}")

    model.to(device)
    model.train()

    print(f"\n[{model_name}] 학습 시작... (Device: {device})")
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                sensor_batch, context_batch = batch
            else:
                # fallback (단일 텐서일 경우)
                sensor_batch = batch
                context_batch = torch.zeros(batch.size(0), batch.size(1), 1)

            sensor_batch = sensor_batch.to(device)
            context_batch = context_batch.to(device)

            optimizer.zero_grad()

            # Conditional Forward: 센서 + 문맥
            output = model(sensor_batch, context_batch)
            # Loss는 센서 복원 오차만 측정 (문맥은 조건일 뿐 복원 대상 아님)
            loss = criterion(output, sensor_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            pbar.set_postfix({"Loss": f"{loss.item():.6f}"})

        avg_loss = total_loss / max(batch_count, 1)
        print(f"[{model_name}] Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        
        # 매 에포크마다 중간 저장
        torch.save(model.state_dict(), save_path)
        print(f"[{model_name}] Epoch {epoch+1} 가중치 저장 완료: {save_path}")

    print(f"[{model_name}] 최종 가중치 저장 완료: {save_path}")


def collect_normal_scores(model, data_loader, device, model_name):
    """정상 데이터에 대한 Reconstruction Error 분포 수집 (POT용)"""
    model.to(device)
    model.eval()
    scores = []

    print(f"\n[{model_name}] 정상 데이터 Anomaly Score 수집 중...")
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"[{model_name}] Score Collection"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                sensor_batch, context_batch = batch
            else:
                sensor_batch = batch
                context_batch = torch.zeros(batch.size(0), batch.size(1), 1)

            sensor_batch = sensor_batch.to(device)
            context_batch = context_batch.to(device)

            output = model(sensor_batch, context_batch)
            # 배치 내 각 윈도우별 MSE
            per_window_mse = torch.mean((output - sensor_batch)**2, dim=(1, 2))
            scores.extend(per_window_mse.cpu().numpy().tolist())

    return np.array(scores)


def compute_pot_threshold(scores, q=0.95, fpr=0.01):
    """
    Peaks Over Threshold (POT) 기반 동적 임계값 산출.
    극단값 이론(EVT)의 Generalized Pareto Distribution(GPD)을 사용.

    Args:
        scores: 정상 데이터의 Anomaly Score 배열
        q: 초기 임계값 분위수 (상위 5%를 극단값으로 간주)
        fpr: 목표 오경보율 (False Positive Rate)
    Returns:
        dict: {yellow_threshold, red_threshold, init_threshold, gpd_params}
    """
    try:
        from scipy.stats import genpareto
    except ImportError:
        print("[Warning] scipy 미설치. 고정 임계값 사용.")
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        return {
            'yellow_threshold': float(mean_s + 2 * std_s),
            'red_threshold': float(mean_s + 3 * std_s),
            'method': 'fallback_zscore'
        }

    # 1) 초기 임계값 (q 분위수)
    init_thresh = np.quantile(scores, q)

    # 2) 초과분(Exceedances) 추출
    exceedances = scores[scores > init_thresh] - init_thresh

    if len(exceedances) < 10:
        print("[Warning] 극단값 샘플 부족. Z-score 기반 fallback 사용.")
        mean_s = np.mean(scores)
        std_s = np.std(scores)
        return {
            'yellow_threshold': float(mean_s + 2 * std_s),
            'red_threshold': float(mean_s + 3 * std_s),
            'method': 'fallback_zscore'
        }

    # 3) GPD 피팅
    c, loc, scale = genpareto.fit(exceedances, floc=0)

    # 4) 동적 임계값 산출 (EVT 공식)
    n = len(scores)
    n_exceed = len(exceedances)
    ratio = n_exceed / n

    # Yellow: FPR=1%에 대응하는 임계값
    yellow_z = init_thresh + (scale / c) * ((fpr / ratio) ** (-c) - 1)
    # Red: FPR=0.1%에 대응하는 임계값
    red_z = init_thresh + (scale / c) * (((fpr / 10) / ratio) ** (-c) - 1)

    print(f"  [POT] init_threshold(q={q}): {init_thresh:.6f}")
    print(f"  [POT] GPD params: c={c:.4f}, scale={scale:.4f}")
    print(f"  [POT] Yellow(FPR={fpr}): {yellow_z:.6f}")
    print(f"  [POT] Red(FPR={fpr/10}): {red_z:.6f}")

    return {
        'yellow_threshold': float(yellow_z),
        'red_threshold': float(red_z),
        'init_threshold': float(init_thresh),
        'gpd_params': {'c': float(c), 'scale': float(scale)},
        'method': 'pot_gpd'
    }


def mock_train(model_name):
    """PyTorch를 사용할 수 없을 때 가상 학습"""
    print(f"\n[{model_name}] Mock(가상) 학습 시작...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}: ", end="")
        for _ in range(10):
            time.sleep(0.1)
            print(".", end="", flush=True)
        mock_loss = 0.05 - (epoch * 0.01) + np.random.uniform(0, 0.005)
        print(f" Avg Loss: {mock_loss:.4f}")

    save_path = os.path.join(WEIGHTS_DIR, f"{model_name.lower()}_mock.dummy")
    with open(save_path, "w") as f:
        f.write("DUMMY_WEIGHTS")
    print(f"[{model_name}] Mock 가중치 저장 완료: {save_path}")


def main():
    prevent_sleep()
    print("=" * 60)
    print("  State-Aware 하이브리드 앙상블 본학습 파이프라인")
    print("=" * 60)

    # ═══════════════════════════════════════════════════════
    # Phase 1: 전처리 파이프라인 피팅
    # ═══════════════════════════════════════════════════════
    print("\n[Phase 1] 전처리 파이프라인 피팅 (RobustScaler + 문맥 인코더)...")
    pipeline = PreprocessingPipeline(weights_dir=WEIGHTS_DIR)
    pipeline.fit_from_files(data_dir=DATA_DIR, chunksize=CHUNKSIZE * 5)

    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))

    # ═══════════════════════════════════════════════════════
    # Phase 2: Conditional 모델 3종 학습
    # ═══════════════════════════════════════════════════════
    if not TORCH_AVAILABLE:
        print("\nPyTorch를 사용할 수 없어 Mock Training으로 전환합니다.")
        mock_train("TR_AE")
        mock_train("ST_GNN")
        mock_train("LSTM_AE")
        print("\n=== 전체 학습 프로세스 완료! (Mock) ===")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Phase 2] Conditional 모델 3종 학습 시작 (Device: {device})")

    num_features = len(SENSOR_FEATURES)
    num_recipe_steps = pipeline.num_recipe_steps
    num_recipes = pipeline.num_recipes
    num_stages = pipeline.num_stages
    num_tools = pipeline.num_tools

    print(f"  센서 변수: {num_features}개")
    print(f"  recipe_step 카테고리: {num_recipe_steps}종")
    print(f"  recipe 카테고리: {num_recipes}종")
    print(f"  stage 카테고리: {num_stages}종")
    print(f"  Tool 카테고리: {num_tools}종")

    # 모델 생성
    model_tr = ConditionalTransformerAE(
        num_features=num_features,
        num_recipe_steps=num_recipe_steps, num_recipes=num_recipes, num_stages=num_stages, num_tools=num_tools,
        embed_dim=8, d_model=32, nhead=4, num_layers=2
    )
    model_gnn = ConditionalSTGNNAE(
        num_features=num_features,
        num_recipe_steps=num_recipe_steps, num_recipes=num_recipes, num_stages=num_stages, num_tools=num_tools,
        embed_dim=8, hidden_dim=16
    )
    model_lstm = ConditionalLSTMAE(
        num_features=num_features,
        num_recipe_steps=num_recipe_steps, num_recipes=num_recipes, num_stages=num_stages, num_tools=num_tools,
        embed_dim=8, hidden_dim=32, num_layers=2
    )

    criterion = nn.MSELoss()
    models = [
        ("TR_AE", model_tr, 0.001),
        ("ST_GNN", model_gnn, 0.001),
        ("LSTM_AE", model_lstm, 0.001),
    ]

    for model_name, model, lr in models:
        # 각 모델마다 새로운 DataLoader를 생성 (IterableDataset은 재사용 불가)
        dataset = StreamingWindowDataset(csv_files, pipeline, seq_len=SEQ_LEN, chunksize=CHUNKSIZE)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_model(model, loader, optimizer, criterion, model_name, device)

    # ═══════════════════════════════════════════════════════
    # Phase 3: POT 동적 임계값 산출
    # ═══════════════════════════════════════════════════════
    print("\n[Phase 3] 정상 데이터 Anomaly Score 분포 수집 → POT 동적 임계값 산출...")

    # 앙상블 점수 수집용 DataLoader (다시 생성)
    score_dataset = StreamingWindowDataset(csv_files, pipeline, seq_len=SEQ_LEN, chunksize=CHUNKSIZE)
    score_loader = torch.utils.data.DataLoader(score_dataset, batch_size=BATCH_SIZE)

    all_ensemble_scores = []
    model_tr.eval()
    model_gnn.eval()
    model_lstm.eval()

    with torch.no_grad():
        for batch in tqdm(score_loader, desc="[Ensemble Score Collection]"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                sensor_batch, context_batch = batch
            else:
                sensor_batch = batch
                context_batch = torch.zeros(batch.size(0), batch.size(1), 1)

            sensor_batch = sensor_batch.to(device)
            context_batch = context_batch.to(device)

            # 3개 모델의 MSE 각각 계산
            out_tr = model_tr(sensor_batch, context_batch)
            out_gnn = model_gnn(sensor_batch, context_batch)
            out_lstm = model_lstm(sensor_batch, context_batch)

            mse_tr = torch.mean((out_tr - sensor_batch)**2, dim=(1, 2))
            mse_gnn = torch.mean((out_gnn - sensor_batch)**2, dim=(1, 2))
            mse_lstm = torch.mean((out_lstm - sensor_batch)**2, dim=(1, 2))

            # 기본 균등 가중합 앙상블 (동적 가중치는 추론 시 적용)
            ensemble = 0.33 * mse_tr + 0.34 * mse_gnn + 0.33 * mse_lstm
            all_ensemble_scores.extend(ensemble.cpu().numpy().tolist())

    all_ensemble_scores = np.array(all_ensemble_scores)
    print(f"  수집된 정상 Anomaly Score 수: {len(all_ensemble_scores):,}개")
    print(f"  평균: {all_ensemble_scores.mean():.6f}, 표준편차: {all_ensemble_scores.std():.6f}")

    # POT 임계값 산출
    pot_result = compute_pot_threshold(all_ensemble_scores, q=0.95, fpr=0.01)

    # 저장
    pot_path = os.path.join(WEIGHTS_DIR, "pot_threshold.pkl")
    joblib.dump(pot_result, pot_path)
    print(f"  POT 임계값 저장 완료: {pot_path}")

    print("\n" + "=" * 60)
    print("  전체 학습 프로세스 완료!")
    print(f"  - 모델 가중치: {WEIGHTS_DIR}/*.pt")
    print(f"  - 스케일러: {WEIGHTS_DIR}/scaler.pkl")
    print(f"  - 문맥 인코더: {WEIGHTS_DIR}/context_encoder.pkl")
    print(f"  - POT 임계값: {WEIGHTS_DIR}/pot_threshold.pkl")
    print("=" * 60)

    allow_sleep()

if __name__ == "__main__":
    main()
