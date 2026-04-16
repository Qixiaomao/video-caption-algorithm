#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Chainlit frontend entry.

No model, torch, or inference imports are allowed here. The only product
runtime path is Chainlit -> REST API -> FastAPI server -> core engine.
"""

import sys
from pathlib import Path

import chainlit as cl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.api_client import infer_caption


def format_caption_result(result: dict) -> str:
    best = result.get("BEST") or {}
    best_key = best.get("key") or "BEST"
    best_text = best.get("text") or "No caption generated."
    candidates = [
        f"S1: {result.get('S1') or '-'}",
        f"S2: {result.get('S2') or '-'}",
        f"S3: {result.get('S3') or '-'}",
    ]
    return (
        "[In-house model result]\n\n"
        f"**Caption:** {best_text}\n\n"
        f"Selected candidate: `{best_key}`\n\n"
        "Candidate captions:\n"
        + "\n".join(f"- {item}" for item in candidates)
    )


@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "Welcome to Video Captioning Demo.\n\n"
            "Please paste an absolute or relative frame directory path, for example:\n"
            "`data/processed/msvd/val/frames/0lh_UWF9ZP4_21_26`"
        ),
        actions=[
            cl.Action(name="engine_resident", value="resident", label="In-house model", payload={}),
        ],
    ).send()


@cl.action_callback("engine_resident")
async def select_resident(action: cl.Action):
    cl.user_session.set("engine", "resident")
    await cl.Message(content="Selected: In-house model").send()


@cl.on_message
async def on_message(message: cl.Message):
    frames_dir = (message.content or "").strip()
    status = cl.Message(content="Validating input...")
    await status.send()

    path = Path(frames_dir)
    if not path.exists() or not path.is_dir():
        status.content = f"The path does not exist or is not a directory:\n`{frames_dir}`"
        await status.update()
        return

    status.content = "Calling FastAPI inference service..."
    await status.update()

    try:
        result = await infer_caption(frames_dir)
        status.content = format_caption_result(result)
        await status.update()
    except Exception as exc:
        status.content = f"Inference request failed:\n```text\n{exc}\n```"
        await status.update()
