"""
Optuna 하이퍼파라미터 튜닝 스크립트
=====================================
3단계 피라미드 접근법으로 시계열 데이터의 문맥을 보존하며
최적의 하이퍼파라미터를 자동 탐색합니다.

Usage:
    # Stage 0: 데이터 캐싱 (먼저 실행)
    python src/data/prepare_tuning_data.py

    # Stage 1: 경량 탐색 (50 trials x 3 models)
    python src/train_optuna.py --stage 1

    # Stage 2: 정밀 탐색 (15 trials x 3 models)
    python src/train_optuna.py --stage 2

    # Stage 3: 최종 본학습 (최적 파라미터로)
    python src/train_optuna.py --stage 3

    # 전체 파이프라인 (Stage 1 → 2 → 3 순차 실행)
    python src/train_optuna.py --stage all

    # Dry run (GPU 확인 + 2 trials만)
    python src/train_optuna.py --stage 1 --n-trials 2 --dry-run
"""
import os
import sys
import glob
import json
import argparse
import time
import ctypes
import numpy as np
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    print("[Warning] optuna 미설치. pip install optuna 실행 필요.")
    OPTUNA_AVAILABLE = False

sys.path.insert(0, os.path.abspath("."))
from src.data.preprocessing import SENSOR_FEATURES, CONTEXT_DIM, PreprocessingPipeline
from src.models.transformer_ae import ConditionalTransformerAE
from src.models.gnn_ae import ConditionalSTGNNAE
from src.models.lstm_ae import ConditionalLSTMAE

# ─── 설정 ──────────────────────────────────────────────────
CACHE_DIR = "data/cached_tensors"
WEIGHTS_DIR = "src/models/weights"
DB_DIR = "data/optuna_db"
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ─── 캐싱된 텐서 데이터셋 ────────────────────────────────────
class CachedTensorDataset(Dataset):
    """캐싱된 .pt 파일들을 메모리에 로드하여 빠르게 접근"""

    def __init__(self, cache_dir, file_list=None):
        """
        Args:
            cache_dir: .pt 파일들이 있는 디렉터리
            file_list: 특정 .pt 파일만 로드 (Run-based CV용). None이면 전부 로드.
        """
        if file_list is not None:
            pt_files = file_list
        else:
            pt_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

        all_sensor = []
        all_context = []

        for pf in pt_files:
            data = torch.load(pf, weights_only=False)
            all_sensor.append(data['sensor'])
            all_context.append(data['context'])

        if all_sensor:
            self.sensor = torch.cat(all_sensor, dim=0)
            self.context = torch.cat(all_context, dim=0)
        else:
            self.sensor = torch.empty(0)
            self.context = torch.empty(0)

    def __len__(self):
        return len(self.sensor)

    def __getitem__(self, idx):
        return self.sensor[idx], self.context[idx]


# ─── 모델 생성 팩토리 ────────────────────────────────────────
def create_model(model_type, meta, params):
    """trial 파라미터로부터 모델 인스턴스를 생성"""
    num_features = meta['num_sensor_features']
    common = {
        'num_features': num_features,
        'num_recipe_steps': meta['num_recipe_steps'],
        'num_recipes': meta['num_recipes'],
        'num_stages': meta['num_stages'],
        'num_tools': meta['num_tools'],
        'embed_dim': params['embed_dim'],
    }

    if model_type == 'TR_AE':
        return ConditionalTransformerAE(
            **common,
            d_model=params['d_model'],
            nhead=params['nhead'],
            num_layers=params['num_layers'],
            dropout=params.get('dropout', 0.1),
        )
    elif model_type == 'ST_GNN':
        return ConditionalSTGNNAE(
            **common,
            hidden_dim=params['hidden_dim'],
        )
    elif model_type == 'LSTM_AE':
        return ConditionalLSTMAE(
            **common,
            hidden_dim=params['hidden_dim'],
            num_layers=params['num_layers'],
            dropout=params.get('dropout', 0.0),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ─── 학습 루프 (AMP 지원) ────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, use_amp=True):
    """1 에포크 학습. AMP(Mixed Precision) 지원."""
    model.train()
    total_loss = 0
    count = 0
    scaler = GradScaler(enabled=use_amp)

    for sensor_batch, context_batch in loader:
        sensor_batch = sensor_batch.to(device)
        context_batch = context_batch.to(device)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            output = model(sensor_batch, context_batch)
            loss = criterion(output, sensor_batch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)


def eval_epoch(model, loader, criterion, device, use_amp=True):
    """검증 에포크"""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for sensor_batch, context_batch in loader:
            sensor_batch = sensor_batch.to(device)
            context_batch = context_batch.to(device)

            with autocast(enabled=use_amp):
                output = model(sensor_batch, context_batch)
                loss = criterion(output, sensor_batch)

            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)


# ─── Stage 1: 경량 탐색 ─────────────────────────────────────
def stage1_objective(trial, model_type, meta, device, n_epochs=1):
    """Stage 1: 넓은 범위 경량 탐색"""
    cache_dir = os.path.join(CACHE_DIR, "stage1_20pct_stride3")
    pt_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    if not pt_files:
        raise RuntimeError(f"캐싱 데이터 없음: {cache_dir}. prepare_tuning_data.py 먼저 실행.")

    # Hold-out 분할 (80/20, Run 단위)
    split_idx = max(1, int(len(pt_files) * 0.8))
    train_files = pt_files[:split_idx]
    val_files = pt_files[split_idx:]

    if not val_files:
        val_files = [train_files.pop()]

    # 공통 파라미터 샘플링
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'embed_dim': trial.suggest_categorical('embed_dim', [4, 8, 16]),
    }

    # 모델별 전용 파라미터
    if model_type == 'TR_AE':
        params['d_model'] = trial.suggest_categorical('d_model', [16, 32, 64, 128])
        params['nhead'] = trial.suggest_categorical('nhead', [2, 4, 8])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 4)
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.2, step=0.1)
        # d_model은 nhead로 나누어 떨어져야 함
        if params['d_model'] % params['nhead'] != 0:
            raise optuna.TrialPruned()

    elif model_type == 'ST_GNN':
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [8, 16, 32, 64])

    elif model_type == 'LSTM_AE':
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [16, 32, 64, 128])
        params['num_layers'] = trial.suggest_int('num_layers', 1, 3)
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.2, step=0.1)

    # 데이터 로드
    train_dataset = CachedTensorDataset(cache_dir, train_files)
    val_dataset = CachedTensorDataset(cache_dir, val_files)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'],
                              shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'],
                            shuffle=False, pin_memory=True, num_workers=2)

    # 모델 생성 및 학습
    model = create_model(model_type, meta, params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.MSELoss()
    use_amp = device.type == 'cuda'

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_amp)
        val_loss = eval_epoch(model, val_loader, criterion, device, use_amp)

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return val_loss


# ─── Stage 2: 정밀 탐색 ─────────────────────────────────────
def stage2_objective(trial, model_type, meta, device, best_params_stage1, n_epochs=2):
    """Stage 2: Stage 1 결과 기반 정밀 탐색 + 2-Fold CV"""
    cache_dir = os.path.join(CACHE_DIR, "stage2_30pct_stride2")
    pt_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    if not pt_files:
        raise RuntimeError(f"캐싱 데이터 없음: {cache_dir}")

    # Stage 1 최적값 주변에서 좁은 범위 탐색
    bp = best_params_stage1

    params = {
        'learning_rate': trial.suggest_float('learning_rate',
            max(1e-5, bp['learning_rate'] * 0.3),
            min(1e-2, bp['learning_rate'] * 3.0), log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'embed_dim': trial.suggest_categorical('embed_dim', [4, 8, 16]),
    }

    if model_type == 'TR_AE':
        # Stage 1 최적값 ± 1단계 범위에서 탐색
        d_model_opts = [16, 32, 64, 128]
        best_d = bp.get('d_model', 32)
        idx = d_model_opts.index(best_d) if best_d in d_model_opts else 1
        narrow = d_model_opts[max(0, idx-1):min(len(d_model_opts), idx+2)]
        params['d_model'] = trial.suggest_categorical('d_model', narrow)
        params['nhead'] = trial.suggest_categorical('nhead', [2, 4, 8])
        params['num_layers'] = trial.suggest_int('num_layers',
            max(1, bp.get('num_layers', 2) - 1),
            min(4, bp.get('num_layers', 2) + 1))
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)
        if params['d_model'] % params['nhead'] != 0:
            raise optuna.TrialPruned()

    elif model_type == 'ST_GNN':
        hidden_opts = [8, 16, 32, 64]
        best_h = bp.get('hidden_dim', 16)
        idx = hidden_opts.index(best_h) if best_h in hidden_opts else 1
        narrow = hidden_opts[max(0, idx-1):min(len(hidden_opts), idx+2)]
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', narrow)

    elif model_type == 'LSTM_AE':
        hidden_opts = [16, 32, 64, 128]
        best_h = bp.get('hidden_dim', 32)
        idx = hidden_opts.index(best_h) if best_h in hidden_opts else 1
        narrow = hidden_opts[max(0, idx-1):min(len(hidden_opts), idx+2)]
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', narrow)
        params['num_layers'] = trial.suggest_int('num_layers',
            max(1, bp.get('num_layers', 2) - 1),
            min(3, bp.get('num_layers', 2) + 1))
        params['dropout'] = trial.suggest_float('dropout', 0.0, 0.3, step=0.05)

    # 2-Fold Cross Validation (Run 단위)
    mid = len(pt_files) // 2
    folds = [
        (pt_files[:mid], pt_files[mid:]),
        (pt_files[mid:], pt_files[:mid]),
    ]

    fold_losses = []
    criterion = nn.MSELoss()
    use_amp = device.type == 'cuda'

    for fold_idx, (train_files, val_files) in enumerate(folds):
        train_dataset = CachedTensorDataset(cache_dir, train_files)
        val_dataset = CachedTensorDataset(cache_dir, val_files)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'],
                                  shuffle=True, pin_memory=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'],
                                shuffle=False, pin_memory=True, num_workers=2)

        model = create_model(model_type, meta, params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

        for epoch in range(n_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device, use_amp)

        val_loss = eval_epoch(model, val_loader, criterion, device, use_amp)
        fold_losses.append(val_loss)

    # Objective: 평균 + 0.5 * 표준편차 (안정성 페널티)
    mean_loss = np.mean(fold_losses)
    std_loss = np.std(fold_losses)
    objective = mean_loss + 0.5 * std_loss

    trial.set_user_attr('mean_val_mse', float(mean_loss))
    trial.set_user_attr('std_val_mse', float(std_loss))

    return objective


# ─── Stage 3: 최종 본학습 ────────────────────────────────────
def stage3_full_training(model_type, meta, device, best_params, n_epochs=3):
    """Stage 2 최적 파라미터로 전체 데이터 본학습"""
    cache_dir = os.path.join(CACHE_DIR, "stage3_full_stride1")
    pt_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    if not pt_files:
        raise RuntimeError(f"캐싱 데이터 없음: {cache_dir}")

    dataset = CachedTensorDataset(cache_dir)
    loader = DataLoader(dataset, batch_size=best_params['batch_size'],
                        shuffle=True, pin_memory=True, num_workers=4)

    model = create_model(model_type, meta, best_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    criterion = nn.MSELoss()
    use_amp = device.type == 'cuda'

    print(f"\n[{model_type}] Stage 3 본학습 시작 ({len(dataset):,}개 윈도우)")
    for epoch in range(n_epochs):
        avg_loss = train_epoch(model, loader, optimizer, criterion, device, use_amp)
        print(f"  [{model_type}] Epoch {epoch+1}/{n_epochs} Avg Loss: {avg_loss:.6f}")

        # 에포크별 체크포인트 저장
        save_path = os.path.join(WEIGHTS_DIR, f"{model_type.lower()}.pt")
        torch.save(model.state_dict(), save_path)
        print(f"  [{model_type}] 체크포인트 저장: {save_path}")

    return model


# ─── POT 임계값 + 앙상블 가중치 ────────────────────────────────
def compute_ensemble_weights_and_pot(models, meta, device, best_params_all):
    """본학습 완료 후 앙상블 가중치 자동 결정 + POT 임계값 산출"""
    cache_dir = os.path.join(CACHE_DIR, "stage3_full_stride1")
    pt_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))

    # 전체 데이터에서 각 모델의 복원 오차 수집
    dataset = CachedTensorDataset(cache_dir)
    loader = DataLoader(dataset, batch_size=256, shuffle=False,
                        pin_memory=True, num_workers=4)

    model_names = ['TR_AE', 'ST_GNN', 'LSTM_AE']
    all_scores = {name: [] for name in model_names}
    all_ensemble = []

    use_amp = device.type == 'cuda'

    for model_name, model in zip(model_names, models):
        model.eval()

    print("\n[앙상블 가중치 + POT] Anomaly Score 수집 중...")
    with torch.no_grad():
        for sensor_batch, context_batch in tqdm(loader, desc="Score Collection"):
            sensor_batch = sensor_batch.to(device)
            context_batch = context_batch.to(device)

            mse_scores = {}
            for model_name, model in zip(model_names, models):
                with autocast(enabled=use_amp):
                    output = model(sensor_batch, context_batch)
                mse = torch.mean((output.float() - sensor_batch.float())**2, dim=(1, 2))
                mse_scores[model_name] = mse
                all_scores[model_name].extend(mse.cpu().numpy().tolist())

            # 균등 앙상블
            ensemble = sum(mse_scores.values()) / len(mse_scores)
            all_ensemble.extend(ensemble.cpu().numpy().tolist())

    # Inverse-Variance Weighting
    print("\n[Inverse-Variance Weighting] 모델별 가중치 자동 결정...")
    variances = {}
    for name in model_names:
        scores = np.array(all_scores[name])
        variances[name] = np.var(scores) + 1e-10  # 0 방지

    inv_vars = {name: 1.0 / var for name, var in variances.items()}
    total_inv = sum(inv_vars.values())
    weights = {name: inv_v / total_inv for name, inv_v in inv_vars.items()}

    for name in model_names:
        print(f"  {name}: var={variances[name]:.6f}, weight={weights[name]:.4f}")

    # POT 임계값 산출
    print("\n[POT] 동적 임계값 산출...")
    ensemble_scores = np.array(all_ensemble)
    try:
        from scipy.stats import genpareto
        q = 0.95
        init_thresh = np.quantile(ensemble_scores, q)
        exceedances = ensemble_scores[ensemble_scores > init_thresh] - init_thresh

        if len(exceedances) >= 10:
            c, loc, scale = genpareto.fit(exceedances, floc=0)
            n = len(ensemble_scores)
            n_exceed = len(exceedances)
            ratio = n_exceed / n
            fpr = 0.01

            yellow_z = init_thresh + (scale / c) * ((fpr / ratio) ** (-c) - 1)
            red_z = init_thresh + (scale / c) * (((fpr / 10) / ratio) ** (-c) - 1)

            pot_result = {
                'yellow_threshold': float(yellow_z),
                'red_threshold': float(red_z),
                'init_threshold': float(init_thresh),
                'gpd_params': {'c': float(c), 'scale': float(scale)},
                'method': 'pot_gpd',
                'ensemble_weights': weights,
            }
            print(f"  Yellow: {yellow_z:.6f}, Red: {red_z:.6f}")
        else:
            raise ValueError("극단값 부족")
    except Exception as e:
        print(f"  [Fallback] GPD 피팅 실패 ({e}). Z-score 기반 임계값 사용.")
        mean_s = np.mean(ensemble_scores)
        std_s = np.std(ensemble_scores)
        pot_result = {
            'yellow_threshold': float(mean_s + 2 * std_s),
            'red_threshold': float(mean_s + 3 * std_s),
            'method': 'fallback_zscore',
            'ensemble_weights': weights,
        }

    pot_path = os.path.join(WEIGHTS_DIR, "pot_threshold.pkl")
    joblib.dump(pot_result, pot_path)
    print(f"  POT + 앙상블 가중치 저장 완료: {pot_path}")

    return pot_result


# ─── 시스템 유틸리티 ──────────────────────────────────────────
def prevent_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        print("[System] 절전 모드 방지 활성화")
    except Exception:
        pass

def allow_sleep():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        print("[System] 절전 모드 방지 해제")
    except Exception:
        pass


# ─── 메인 ───────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Optuna 하이퍼파라미터 튜닝")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["1", "2", "3", "all"],
                        help="실행할 Stage (1/2/3/all)")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="트라이얼 수 (기본: Stage1=50, Stage2=15)")
    parser.add_argument("--dry-run", action="store_true",
                        help="GPU 확인 + 2 trials만 실행")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="에포크 수 오버라이드")
    args = parser.parse_args()

    if not OPTUNA_AVAILABLE:
        print("[Error] optuna를 설치해주세요: pip install optuna")
        return

    prevent_sleep()

    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"\n🎮 GPU 감지: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        print("\n⚠️ GPU 미감지. CPU로 실행합니다. (매우 느림)")

    # 메타 데이터 로드
    meta_path = os.path.join(CACHE_DIR, "meta.pt")
    if not os.path.exists(meta_path):
        print(f"[Error] {meta_path} 없음. 먼저 실행: python src/data/prepare_tuning_data.py")
        return
    meta = torch.load(meta_path, weights_only=False)

    print(f"\n📊 데이터셋 정보:")
    print(f"   센서 변수: {meta['num_sensor_features']}개")
    print(f"   recipe_step: {meta['num_recipe_steps']}종")
    print(f"   recipe: {meta['num_recipes']}종")
    print(f"   stage: {meta['num_stages']}종")

    model_types = ['TR_AE', 'ST_GNN', 'LSTM_AE']
    best_params_all = {}
    stages_to_run = ['1', '2', '3'] if args.stage == 'all' else [args.stage]

    # ═══════════════════════════════════════════════════════
    # Stage 1: 경량 탐색
    # ═══════════════════════════════════════════════════════
    if '1' in stages_to_run:
        print("\n" + "=" * 60)
        print("  Stage 1: 경량 탐색 (Coarse Search)")
        print("=" * 60)

        n_trials_s1 = args.n_trials or (2 if args.dry_run else 50)
        n_epochs_s1 = args.n_epochs or 1

        for model_type in model_types:
            study_name = f"stage1_{model_type}"
            db_path = os.path.join(DB_DIR, f"{study_name}.db")
            storage = f"sqlite:///{db_path}"

            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                direction="minimize",
                sampler=TPESampler(seed=42),
                pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            )

            print(f"\n[{model_type}] Stage 1 시작 ({n_trials_s1} trials, {n_epochs_s1} epochs)")
            start_time = time.time()

            study.optimize(
                lambda trial: stage1_objective(trial, model_type, meta, device, n_epochs_s1),
                n_trials=n_trials_s1,
                show_progress_bar=True,
            )

            elapsed = time.time() - start_time
            print(f"  [{model_type}] 완료! ({elapsed/60:.1f}분)")
            print(f"  최적 val_mse: {study.best_value:.6f}")
            print(f"  최적 파라미터: {study.best_params}")

            best_params_all[model_type] = study.best_params

    # ═══════════════════════════════════════════════════════
    # Stage 2: 정밀 탐색
    # ═══════════════════════════════════════════════════════
    if '2' in stages_to_run:
        print("\n" + "=" * 60)
        print("  Stage 2: 정밀 탐색 (Fine Search)")
        print("=" * 60)

        n_trials_s2 = args.n_trials or (2 if args.dry_run else 15)
        n_epochs_s2 = args.n_epochs or 2

        # Stage 1 결과 로드 (이전 실행에서 저장된 것)
        if not best_params_all:
            for model_type in model_types:
                db_path = os.path.join(DB_DIR, f"stage1_{model_type}.db")
                if os.path.exists(db_path):
                    study = optuna.load_study(
                        study_name=f"stage1_{model_type}",
                        storage=f"sqlite:///{db_path}",
                    )
                    best_params_all[model_type] = study.best_params
                    print(f"  [{model_type}] Stage 1 최적 파라미터 로드: {study.best_params}")

        for model_type in model_types:
            if model_type not in best_params_all:
                print(f"  [{model_type}] Stage 1 결과 없음. Stage 1을 먼저 실행하세요.")
                continue

            study_name = f"stage2_{model_type}"
            db_path = os.path.join(DB_DIR, f"{study_name}.db")
            storage = f"sqlite:///{db_path}"

            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                direction="minimize",
                sampler=TPESampler(seed=42),
            )

            bp = best_params_all[model_type]
            print(f"\n[{model_type}] Stage 2 시작 ({n_trials_s2} trials, {n_epochs_s2} epochs, 2-Fold CV)")
            start_time = time.time()

            study.optimize(
                lambda trial: stage2_objective(trial, model_type, meta, device, bp, n_epochs_s2),
                n_trials=n_trials_s2,
                show_progress_bar=True,
            )

            elapsed = time.time() - start_time
            print(f"  [{model_type}] 완료! ({elapsed/60:.1f}분)")
            print(f"  최적 objective: {study.best_value:.6f}")
            if study.best_trial.user_attrs:
                print(f"  mean_val_mse: {study.best_trial.user_attrs.get('mean_val_mse', 'N/A')}")
                print(f"  std_val_mse: {study.best_trial.user_attrs.get('std_val_mse', 'N/A')}")
            print(f"  최적 파라미터: {study.best_params}")

            best_params_all[model_type] = study.best_params

    # ═══════════════════════════════════════════════════════
    # Stage 3: 최종 본학습
    # ═══════════════════════════════════════════════════════
    if '3' in stages_to_run:
        print("\n" + "=" * 60)
        print("  Stage 3: 최종 본학습 (Full Training)")
        print("=" * 60)

        n_epochs_s3 = args.n_epochs or 3

        # Stage 2 결과 로드
        if not best_params_all:
            for model_type in model_types:
                # Stage 2 먼저 시도, 없으면 Stage 1
                for stage in ['stage2', 'stage1']:
                    db_path = os.path.join(DB_DIR, f"{stage}_{model_type}.db")
                    if os.path.exists(db_path):
                        study = optuna.load_study(
                            study_name=f"{stage}_{model_type}",
                            storage=f"sqlite:///{db_path}",
                        )
                        best_params_all[model_type] = study.best_params
                        print(f"  [{model_type}] {stage} 최적 파라미터 로드 완료")
                        break

        trained_models = []
        for model_type in model_types:
            if model_type not in best_params_all:
                print(f"  [{model_type}] 최적 파라미터 없음. Stage 1/2를 먼저 실행하세요.")
                continue

            bp = best_params_all[model_type]
            print(f"\n[{model_type}] 최적 파라미터: {bp}")

            model = stage3_full_training(model_type, meta, device, bp, n_epochs_s3)
            trained_models.append(model)

        # POT + 앙상블 가중치
        if len(trained_models) == 3:
            compute_ensemble_weights_and_pot(trained_models, meta, device, best_params_all)

    # 최적 파라미터 요약 저장
    summary_path = os.path.join(WEIGHTS_DIR, "best_params.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(best_params_all, f, indent=2, ensure_ascii=False)
    print(f"\n📋 최적 파라미터 요약 저장: {summary_path}")

    print("\n" + "=" * 60)
    print("  하이퍼파라미터 튜닝 완료!")
    print("=" * 60)

    allow_sleep()


if __name__ == "__main__":
    main()
