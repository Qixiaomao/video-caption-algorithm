#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import shlex
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Video Caption Backend", version="1.0.0")

# 允许本地前端直接调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 如需收紧可改为 ["http://localhost:7860"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Request/Response schema
# -----------------------------
class InferRequest(BaseModel):
    frames_dir: str = Field(..., description="包含 frame_*.jpg 的目录")
    ckpt: str = Field(..., description="模型权重 .pt 路径（mapper 微调后的）")
    stage: str = Field("all", description="固定用 all（兼容旧脚本）")
    prefix_len: int = 4
    num_frames: int = 8
    image_size: int = 224
    ln_scale: float = 0.6
    in_weight: float = 0.4
    preset1: str = "precise"
    preset2: str = "precise"
    preset3: str = "natural"
    prompt1: str = ""   # Windows 上如果带引号容易转义混乱，建议留空或简单英文
    prompt2: str = "State the main action in one short sentence:"
    prompt3: str = "Write a short, natural caption:"
    emit_json: bool = True  # 必须为 True，这样 inference 会输出 JSON

class InferResponse(BaseModel):
    S1: str
    S2: str
    S3: str
    BEST: dict

# -----------------------------
# Helpers
# -----------------------------
def _python_executable() -> str:
    # 保证用当前环境的 python 运行 inference 模块
    return sys.executable or "python"

def _build_cmd(req: InferRequest) -> list[str]:
    """
    拼出：python -m inference ... --emit_json
    全部通过参数传入，避免导入签名不一致问题。
    """
    cmd = [
        _python_executable(), "-m", "inference",
        "--frames_dir", req.frames_dir,
        "--stage", req.stage,
        "--ckpt", req.ckpt,
        "--prefix_len", str(req.prefix_len),
        "--num_frames", str(req.num_frames),
        "--image_size", str(req.image_size),
        "--ln_scale", str(req.ln_scale),
        "--in_weight", str(req.in_weight),
        "--preset1", req.preset1,
        "--preset2", req.preset2,
        "--preset3", req.preset3,
        "--prompt1", req.prompt1 if req.prompt1 else " ",
        "--prompt2", req.prompt2 if req.prompt2 else " ",
        "--prompt3", req.prompt3 if req.prompt3 else " ",
        "--emit_json"
    ]
    return cmd

def _run_inference_subprocess(req: InferRequest) -> dict:
    """
    以子进程方式调用你现有的 inference.py（模块方式 -m inference）。
    读取 stdout 最后一条 JSON 行。
    """
    # 兼容有空格的路径（比如 “INTI courses”）
    cmd = _build_cmd(req)

    # 确保当前工作目录就是项目根（inference.py 所在处）
    # 如果 fastapi_app.py 就在项目根，这里用文件所在目录即可
    project_root = Path(__file__).resolve().parent

    # 给 Python 找到 src 包
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)

    try:
        proc = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,  # 我们自己解析返回码
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to launch inference: {e}")

    if proc.returncode != 0:
        # 把关键日志带回去排障
        raise HTTPException(
            status_code=500,
            detail=f"inference failed (code={proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # 从 stdout 里找到最后一个 JSON 行
    stdout_lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    json_obj = None
    for ln in reversed(stdout_lines):
        if ln.startswith("{") and ln.endswith("}"):
            try:
                json_obj = json.loads(ln)
                break
            except json.JSONDecodeError:
                continue
    if not json_obj:
        # 把 stdout 打印回去便于快速检查
        raise HTTPException(
            status_code=500,
            detail=f"cannot find JSON in inference output.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # 期望字段：S1/S2/S3/BEST
    for k in ("S1", "S2", "S3", "BEST"):
        if k not in json_obj:
            raise HTTPException(status_code=500, detail=f"bad JSON shape from inference (missing {k}). JSON={json_obj}")

    return json_obj

# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    # 基础校验
    frames = Path(req.frames_dir)
    ckpt = Path(req.ckpt)
    if not frames.exists():
        raise HTTPException(status_code=400, detail=f"frames_dir not found: {frames}")
    if not ckpt.exists():
        raise HTTPException(status_code=400, detail=f"ckpt not found: {ckpt}")

    data = _run_inference_subprocess(req)
    return data