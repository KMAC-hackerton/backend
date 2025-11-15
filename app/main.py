from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import numpy as np
import random

import config
from app.controller import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    print("[lifespan] Application started")
    yield
    print("[lifespan] Shutdown finishing")


app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Cost-Summary"],
)

app.include_router(api_router)