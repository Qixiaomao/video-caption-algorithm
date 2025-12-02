#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chainlit 前端入口（稳健版）
- 兼容链路：
  🔧 自研推理 -> inference.run_one_video(...)
  🤖 备用模型 -> 优先调用 tools.caption_fallback_pt.caption_vitgpt2_from_frames_dir
               若不存在该函数，则回退到 caption_blip_from_frames_dir（仅在 torch>=2.6 或模型有 safetensors 时可用）
- 适配点：
  * 不使用 ActionList（部分 Chainlit 版本无）
  * Message.update() 正确用法：先改 content，再 await update()
  * 校验 frames_dir 是否存在
  * 禁止 transformers 触发 TensorFlow 导入
"""

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"   # 禁 TF
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import sys
import traceback
from pathlib import Path
import chainlit as cl

# 让 Python 能找到仓库根目录里的 inference.py / tools/*
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# ========== 会话开始 ==========
@cl.on_chat_start
async def on_start():
    await cl.Message(
        content=(
            "🎬 Weclome to Video Subtitles Demo！\n\n"
            "Please select an inference engine first then paste into the input box **Absolute or relative path to the frame directory**（For example: \n"
            "`data/processed/msvd/val/frames/0lh_UWF9ZP4_21_26`）。"
        ),
        actions=[
            cl.Action(
                name="engine_resident",
                value="resident",
                label="🔧 In-house reasoning",
                payload={}              # ✅ 必填：空字典也行
            ),
            cl.Action(
                name="engine_blip",
                value="blip",
                label="🤖 BLIP Standby",
                payload={}              # ✅ 必填：空字典也行
            ),
        ],
    ).send()


# ========== 引擎选择回调 ==========
@cl.action_callback("engine_resident")
async def select_resident(action: cl.Action):
    cl.user_session.set("engine", "resident")
    await cl.Message(content="✅ Selected：In-house reasoning").send()


@cl.action_callback("engine_blip")
async def select_blip(action: cl.Action):
    cl.user_session.set("engine", "blip")
    await cl.Message(content="✅ Selected：BLIP Standby").send()


# ========== 主处理 ==========
@cl.on_message
async def on_message(message: cl.Message):
    engine = cl.user_session.get("engine") or "resident"
    frames_dir = (message.content or "").strip()

    # 统一的“状态消息”：先发，再逐步 update()
    status = cl.Message(content="⏳ Input is being validated…")
    await status.send()

    # 1) 路径检查
    p = Path(frames_dir)
    if not p.exists() or not p.is_dir():
        status.content = f"❌ The path does not exist or is not a directory：\n`{frames_dir}`"
        await status.update()
        return

    # 2) 打印选择与基本信息
    status.content = (
        f"🛠️ Engine：{'In-house' if engine=='resident' else 'BLIP'}\n"
        f"📁 List：`{frames_dir}`\n"
        f"▶️ Start reasoning…"
    )
    await status.update()

    try:
        if engine == "resident":
            # —— 自研推理 —— #
            from inference import run_one_video

            out = run_one_video(
                frames_dir=frames_dir,
                ckpt=r".\checkpoints\msvd_mapper_finetune_v2.pt",
                stage="all",
                prefix_len=4,
                num_frames=8,
                image_size=224,
                ln_scale=0.6,
                in_weight=0.4,
                preset1="precise",
                preset2="precise",
                preset3="natural",
                emit_json=True,
            )
            text = out if isinstance(out, str) else str(out)
            status.content = f"✅ [In-house research results]\n{text}"
            await status.update()

        else:
            # —— 备用模型（优先 vit-gpt2，有 safetensors，不触发 torch>=2.6 限制）—— #
            caption = None
            err_msgs = []

            try:
                # 优先：vit-gpt2（最稳，不触发 torch.load 漏洞限制）
                from tools.caption_fallback_pt import caption_vitgpt2_from_frames_dir

                status.content = "🤖 Generating with vit-gpt2 (security weights)…"
                await status.update()

                caption = caption_vitgpt2_from_frames_dir(
                    frames_dir=frames_dir,
                    num_frames=8,
                    model="nlpconnect/vit-gpt2-image-captioning",
                )
            except Exception as e_vit:
                err_msgs.append(f"[vit-gpt2] {e_vit}")

            if caption is None:
                # 退路：BLIP（只有在模型仓库提供 safetensors 或你本地 torch>=2.6 时可靠）
                try:
                    from tools.caption_fallback_pt import caption_blip_from_frames_dir

                    status.content = "🤖 vit-gpt2 failed, reverted to BLIP…"
                    await status.update()

                    # 若已升级 torch>=2.6，可使用 BLIP；否则可能再被安全限制拦住
                    caption = caption_blip_from_frames_dir(
                        frames_dir=frames_dir,
                        num_frames=8,
                        model="Salesforce/blip-image-captioning-base",
                    )
                except Exception as e_blip:
                    err_msgs.append(f"[blip] {e_blip}")

            if caption is None:
                # 两条都失败 -> 汇总错误
                msg = " / ".join(err_msgs) if err_msgs else "unknown error"
                status.content = f"❌ The backup model also failed:{msg}\n\nRecommendation: Prioritize using the vit-gpt2 model or upgrade to torch version 2.6 or higher."
                await status.update()
            else:
                status.content = f"✅ [Backup Model Results]\n{caption}"
                await status.update()

    except TypeError:
        tb = traceback.format_exc()
        status.content = f"❌ Parameter error:\n```\n{tb}\n```"
        await status.update()
    except Exception as e:
        tb = traceback.format_exc()
        status.content = f"❌ Failed to run: {e}\n```\n{tb}\n```"
        await status.update()