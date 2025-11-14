import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTDIR = BASE_DIR / "outputs"
OUTDIR.mkdir(exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 11
GRID_T = 123
GRID_Y = 40
GRID_X = 40
VSET = (8, 10, 12, 14, 16)
KAPPA_UNCERT = 0.5
MIX_PHYS_AND_DL = 0.0

LAT_MIN = 60.0
LAT_MAX = 90.0
LON_MIN = -180.0
LON_MAX = 180.0

MODEL_SAVE_PATH = OUTDIR / "ncf_model_weights.pth"
ENV_DATA_PATH = OUTDIR / "env_fields.npz"
DEFAULT_VIS_PATH = OUTDIR / "route_result.png"
