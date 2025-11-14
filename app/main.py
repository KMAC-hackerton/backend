from fastapi import FastAPI
from contextlib import asynccontextmanager
import numpy as np
import random

import config
from app.controller import api_router
from app.services import Service


@asynccontextmanager
async def lifespan(app: FastAPI):
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    try:
        service = Service()
    except Exception as exc:
        print(f"[lifespan] Failed to initialize Service: {exc}")
        raise
    app.state.service = service
    print("[lifespan] Service initialized")
    yield
    print("[lifespan] Shutdown finishing")


app = FastAPI(
    lifespan=lifespan,
)

app.include_router(api_router)