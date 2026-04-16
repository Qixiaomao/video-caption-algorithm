from __future__ import annotations

import os
from typing import Any, Dict

import httpx

BACKEND_URL = os.getenv("VIDEO_CAPTION_BACKEND_URL", "http://127.0.0.1:8001")


async def infer_caption(frames_dir: str) -> Dict[str, Any]:
    """Frontend-only REST client for video caption inference."""

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=5.0)) as client:
        response = await client.post(f"{BACKEND_URL}/api/v1/infer", json={"frames_dir": frames_dir})
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("detail", detail)
            except Exception:
                pass
            raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}") from exc
        return response.json()
