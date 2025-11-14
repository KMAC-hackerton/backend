import os
import torch
from dataclasses import dataclass
from datetime import datetime

# -------------------- 전역 설정 --------------------
SEED = 11
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Colab 학습 스크립트와 일치해야 하는 값
DATE_START = datetime(2023, 8, 1)
DATE_END = datetime(2023, 12, 1)
# ⚠️ EC2 t3.micro 최적화: 격자 해상도를 줄여 A* 성능 개선
# 원본: (DATE_END - DATE_START).days + 1, 40, 40
GRID_T, GRID_Y, GRID_X = (DATE_END - DATE_START).days + 1, 20, 20  # 40→20 (1/4 감소)
VSET = (10, 14)  # 5개→2개로 줄여 추론 속도 2.5배 향상
KAPPA_UNCERT = 0.5
MIX_PHYS_AND_DL = 0.0

# 지도 범위 (시각화 시 사용)
LAT_MIN, LAT_MAX = 60.0, 90.0
LON_MIN, LON_MAX = -180.0, 180.0

# 디렉토리 경로
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(OUTDIR, "ncf_model_weights.pth")
ENV_DATA_PATH = os.path.join(OUTDIR, "env_fields.npz")
DEFAULT_SAVE_PATH = os.path.join(OUTDIR, "route_result.png")

# -------------------- 📦 컨테이너 및 코스트 정의 ---------------------
@dataclass
class EnvFields:
    SIC: "np.ndarray"; Hs: "np.ndarray"; U10: "np.ndarray"
    dist_ice: "np.ndarray"; depth: "np.ndarray"; bio_mask: "np.ndarray"
    risk_grad_ice: "np.ndarray"; forbid_mask: "np.ndarray"
    
@dataclass
class VesselSpec:
    name: str = "Generic"; fuel_type: str = "MGO"
    alpha_fuel: float = 1.0; v_min: float = 8.0; v_max: float = 16.0

EF_CO2 = {"MGO": 3.206, "LSFO": 3.114, "LNG": 2.75, "eMeOH": 0.25}
EF_BC = {"MGO": 2.5e-3, "LSFO": 3.5e-3, "LNG": 0.7e-3, "eMeOH": 0.3e-3}

@dataclass
class CostWeights:
    w_fuel: float = 0.6; w_bc: float = 0.25; w_noise: float = 0.1; w_risk: float = 0.05
    bigM: float = 1e6

@dataclass
class CostParams:
    H0: float = 2.0; a_s: float = 0.6
    s1: float = 0.15; beta1: float = 8.0; a_i: float = 1.0
    gamma1: float = 8.0; gamma2: float = 0.02
    s_c: float = 0.35; kappa1: float = 10.0; Hs_c: float = 4.0; g_c: float = 0.55
    a0: float = 150.0; a2: float = 0.02; p: float = 3.2