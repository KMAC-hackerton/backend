from fastapi import FastAPI
from contextlib import asynccontextmanager
import torch
import numpy as np
import random

import config
from utils import load_processed_env_fields
from models import TinyNCF, PhysicalCost, NeuralCostFieldDL
from models import VesselSpec, CostWeights, CostParams
from app.controller import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server startup: Initializing...")
    try:
        # 0. 시드 설정
        np.random.seed(config.SEED)
        random.seed(config.SEED)
        torch.manual_seed(config.SEED)

        # 1. (중요) 사전 처리된 환경 데이터 로드
        print("Loading Processed Environment Fields...")
        F = load_processed_env_fields()
        app.state.env_fields = F
        print(f"Environment loaded: T={F.SIC.shape[0]}")

        # 2. (중요) 사전 학습된 NCF 모델 로드
        print("Loading NCF Model...")
        ncf_model = TinyNCF(in_dim=11).to(config.device)
        ncf_model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.device))
        ncf_model.eval()
        app.state.ncf_torch_model = ncf_model

        # 3. 비용 엔진 초기화
        print("Initializing Cost Engine...")
        phys_cost = PhysicalCost(VesselSpec(), CostWeights(), CostParams())
        ncf_cost_model = NeuralCostFieldDL(
            ncf_model, phys=phys_cost, 
            mix=config.MIX_PHYS_AND_DL, 
            kappa=config.KAPPA_UNCERT
        )
        app.state.phys_cost = phys_cost
        app.state.ncf_cost_model = ncf_cost_model # 최종 비용 모델

        print("🟢 Router components loaded successfully.")
    
    except Exception as e:
        print(f"🔴 FATAL: Failed to initialize components: {e}")
        app.state.error = str(e)

    yield

    print("Server shutdown: Cleaning up...")
    app.state.clear()

app = FastAPI(
    title="Arctic Eco-Routing API (Controller-Service)",
    description="의존성 주입을 활용한 컨트롤러-서비스 계층 구조",
    version="2.0",
    lifespan=lifespan
)

app.include_router(api_router)