from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from server.routes.health import router as health_router
from server.routes.inference import router as inference_router
from server.settings import SETTINGS

app = FastAPI(title="Video Caption Backend", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compatibility routes: /health and /infer.
app.include_router(health_router)
app.include_router(inference_router)

# Versioned routes for the product layout: /api/v1/health and /api/v1/infer.
app.include_router(health_router, prefix=SETTINGS.api_prefix)
app.include_router(inference_router, prefix=SETTINGS.api_prefix)

