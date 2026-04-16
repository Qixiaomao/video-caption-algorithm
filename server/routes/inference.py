from fastapi import APIRouter, HTTPException

from server.schemas import InferRequest, InferResponse
from server.services.inference_service import INFERENCE_SERVICE

router = APIRouter(tags=["inference"])


@router.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    try:
        return INFERENCE_SERVICE.infer(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"inference failed: {exc}") from exc

