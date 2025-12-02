#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, logging, os, re, sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# 你项目内的模型类
from src.models.caption_model import VideoCaptionModel


# ------------------------- 基础设施 -------------------------
def setup_logging(level="INFO"):
    level = level.upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
log = logging.getLogger(__name__)


def list_frames(frames_dir: Path):
    files = sorted(frames_dir.glob("frame_*.jpg"))
    return files


def load_frames(frames_dir: Path, num_frames=8, image_size=224, device="cpu"):
    files = list_frames(frames_dir)
    if not files:
        raise SystemExit(f"[FATAL] no frames under {frames_dir}")

    step = max(len(files) // num_frames, 1)
    picks = files[::step][:num_frames]

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    imgs = []
    for p in picks:
        with Image.open(p) as im:
            imgs.append(tfm(im.convert("RGB")))
    video = torch.stack(imgs, dim=0).unsqueeze(0).to(device)  # [1,T,3,224,224]

    log.info(f"frames total={len(files)} | sampled={len(picks)} | picks={[p.name for p in picks[:4]]}")
    return video


# ------------------------- 文本清洗工具 -------------------------
def _strip_acronyms_and_countries(s: str) -> str:
    s = re.sub(r"\bU\.S\.A?\.?\b", "", s, flags=re.I)
    s = re.sub(r"\bUSA\b", "", s, flags=re.I)
    s = re.sub(r"\bUnited States of America\b", "", s, flags=re.I)
    s = re.sub(r"\bUnited States\b", "", s, flags=re.I)
    s = re.sub(r"\bAmerica\b", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _collapse_prep_chain(s: str) -> str:
    # 压缩介词链：in the front of / in front of / in the middle of ... 等
    s = re.sub(r"(?i)\bin\s+the\s+front\s+of\b", "in front of", s)
    s = re.sub(r"(?i)\bin\s+the\s+middle\s+of\b", "in the middle of", s)
    s = re.sub(r"(?i)\bat\s+the\s+side\s+of\b", "at the side of", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _ensure_sit_complement(s: str) -> str:
    """
    若句子只有 'Someone is sitting' 或坐姿缺少地点/宾语，补一个温和安全的补语。
    遇到已经是 'Someone is ...' 的完整句子则不再拼接，避免重复。
    """
    low = s.strip().lower()

    # 已有自然的 'someone is ...'，不再改写
    if re.match(r"^someone\s+is\b", low):
        return s

    # 只有 'someone is sitting'（无补语）
    if re.match(r"^someone\s+is\s+sitting\s*\.?$", low):
        return "Someone is sitting on a chair."

    # sitting 开头但没介词补语，拼一个
    if re.match(r"^someone\s+is\s+sitting\b", low) and not re.search(r"\b(in|on|at|by|with|near)\b", low):
        return s.rstrip(". ") + " on a chair."
    return s


def _truncate_on_noise(s: str) -> str:
    """
    一旦句子里出现明显噪声 token（含数字/斜杠/大写缩写/连字符缩写等），
    立刻从该 token 之前截断，保证结尾有句号。
    """
    if not s:
        return s
    toks = s.split()
    cut = len(toks)

    for i, t in enumerate(toks):
        t_raw = t.strip(",.;:!?()[]{}\"'`")
        if not t_raw:
            continue
        if re.search(r"[0-9/\\]", t_raw):                 # 含数字/斜杠
            cut = i; break
        if re.match(r"^(?:[A-Za-z]\.){2,}$", t_raw):      # A.B. / I.D. / U.S.
            cut = i; break
        if re.match(r"^[A-Z]{1,3}-[A-Za-z0-9]{1,6}$", t_raw):  # W-8 / 1099-MISC
            cut = i; break
        if len(t_raw) <= 3 and t_raw.isupper():           # 孤立大写缩写
            cut = i; break

    toks = toks[:cut] if cut < len(toks) else toks
    s2 = " ".join(toks).strip()
    if s2 and s2[-1] not in ".!?":
        s2 += "."
    return s2


def _prune_weird_tails(s: str) -> str:
    s = re.sub(r"(?i)\b(?:how|why|what|that|which)\b.*$", "", s).strip()
    s = re.sub(r"(?i)\bA\s+wonders\b.*$", "", s).strip()
    if not s:
        return "Someone is in the scene."
    return s


def _dedup_tokens(s: str) -> str:
    # 连续重复词去重（注意 raw \1）
    s = re.sub(r"(?i)\b(\w+)\b(?:\s+\1\b)+", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _ensure_period_and_caps(s: str) -> str:
    s = s.strip()
    if s and s[0].isalpha():
        s = s[0].upper() + s[1:]
    if s and s[-1] not in ".!?":
        s += "."
    return s


def _first_sentence(s: str) -> str:
    parts = re.split(r"\s*(?<=\.|\!|\?)\s+", s)
    return parts[0].strip() if parts and parts[0].strip() else s.strip()


def _score_sentence(s: str) -> float:
    if not s:
        return -1e9
    toks = s.split()
    n = len(toks)
    score = 0.0
    mu, sigma = 12.0, 4.0
    score += -((n - mu) ** 2) / (2 * sigma * sigma)
    if re.search(r"\b\w+ing\b", s): score += 1.0
    if re.search(r"\b(?:is|are|was|were)\b", s): score += 0.5
    if s.endswith((".", "!", "?")): score += 0.3
    if re.search(r"\b(?:[A-Z]\.){2,}\b", s): score -= 1.5
    if re.search(r"(?i)\b(click here|subscribe|report abuse|sign up|pastebin)\b", s): score -= 1.5
    if n < 4: score -= 2.0
    if s.strip().lower() in {"someone is sitting.", "someone is in the scene."}: score -= 0.8
    return score


def clean_text(raw: str) -> str:
    s = (raw or "").strip()
    
    # —— 强一点的网页腔与破折号清理（只干掉噪声，不碰正常句）——
    # 0) 全是破折号/下划线/装饰线：置空
    if re.fullmatch(r'[-—_=\s]{6,}\.?', s):
        return ""

    # 1) 行首破折号/下划线 + 空格：去掉（很多网页模板喜欢这样开头）
    s = re.sub(r'^\s*[-—_=\s]{2,}\s*', "", s)

    # 2) 纯链接/HTML/版权/引号鸡汤：直接置空
    if re.match(r'^\s*(https?://|www\.|<a\b|&lt;a\b)', s, re.I) \
    or re.match(r'^\s*(©|copyright\b)', s, re.I) \
    or re.fullmatch(r'"\s*[^"]+\s*"\.?', s):
        return ""

    # 3) 常见“营销/即将播放/点击”模板句触发：置空
    BAD_LEADS = (
        r"you are about to\b", r"click here\b", r"subscribe\b",
        r"available on youtube\b", r"watch live\b", r"find out\b",
        r"the video will\b", r"on the road\b"
    )
    if re.match(r'^\s*(?:' + "|".join(BAD_LEADS) + r')', s, re.I):
        return ""

    # 4) 如果包含明显 HTML 残片/社交站名：置空
    if re.search(r'(</?\w+>|reddit\.com|pastebin|mailto:)', s, re.I):
        return ""
        
    # —— 细化清理步骤 ——

    # 0) 快速过滤网页腔 & 记录标记
    flagged = bool(re.search(r"(?i)\b(click here|subscribe|report abuse|pastebin|official facebook|video will be)\b", s))
    s = re.sub(r"(?i)\b(click here|subscribe|report abuse|pastebin|official facebook|video will be.*)$", "", s).strip()

    # 1) 去国家缩写/网页尾巴
    s = _strip_acronyms_and_countries(s)

    # 2) 介词链压缩
    s = _collapse_prep_chain(s)

    # 3) 遇到噪声 token 直接截断
    if len(s.split()) >= 10:
        s = _truncate_on_noise(s)

    # 4) 剪掉奇怪尾巴
    s = _prune_weird_tails(s)

    # 5) 若网页腔清空/过短，给个安全兜底（在坐姿补语前后都可，这里放前）
    if flagged and len(s.split()) <= 2:
        s = "Someone is in the scene."

    # 6) 坐姿补语（仅在句式需要时触发）
    s = _ensure_sit_complement(s)

    # 7) 去重、首字母/句末标点
    s = _dedup_tokens(s)
    s = _ensure_period_and_caps(s)

    # 8) 若多句，选分最高的一句
    parts = re.split(r"\s*(?<=\.|\!|\?)\s+", s)
    parts = [t.strip() for t in parts if t.strip()]
    if len(parts) > 1:
        s = max(parts, key=_score_sentence)

    # 9) 兜底：第一句
    s = _first_sentence(s)
    return s


# ------------------------- 生成 -------------------------
def preset_to_kwargs(name: str):
    name = (name or "precise").lower()
    if name == "precise":
        return dict(num_beams=3, max_new_tokens=24, temperature=1.0,
                    top_p=1.0, no_repeat_ngram_size=3, repetition_penalty=1.1)
    if name == "detailed":
        return dict(num_beams=4, max_new_tokens=40, temperature=1.0, # max_new_tokens 更大 32->40
                    top_p=1.0, no_repeat_ngram_size=3, repetition_penalty=1.1)
    if name == "natural":
        return dict(num_beams=1, max_new_tokens=24, temperature=0.9,
                    top_p=0.9, no_repeat_ngram_size=3, repetition_penalty=1.05)
    if name == "safe_sample":
        return dict(num_beams=1, max_new_tokens=22, temperature=0.8,
                    top_p=0.85, no_repeat_ngram_size=3, repetition_penalty=1.1)
    return preset_to_kwargs("precise")


@torch.no_grad()
def generate_once(model: VideoCaptionModel,
                  video: torch.Tensor,
                  prompt: str,
                  ln_scale: float,
                  in_weight: float,
                  **decode_kwargs) -> str:
    device = next(model.parameters()).device
    model.eval()

    # 编码 + prefix 注入（支持 mapper）
    emb = model.encoder(video)            # [B,D] 或 [B,*,D]
    emb = model.proj(emb)                 # -> prefix 空间（通常 [B,P,Dp] 或 [B,1,Dp]）
    # 若有 mapper：decoder 内部会处理；我们这里只做轻量归一化缩放
    if emb.dim() == 2:
        emb = emb.unsqueeze(1)            # [B,1,Dp]

    # 轻量注入（按你训练时的尺度）
    if ln_scale is not None and ln_scale > 0:
        emb = torch.nn.functional.layer_norm(emb, emb.shape[-1:]) * ln_scale
    if in_weight is not None and in_weight > 0:
        emb = emb * in_weight

    # 生成（不传 do_sample/logits_processor 避免接口不匹配）
    text = model.decoder.generate(
        emb,
        prompt=prompt or "",
        max_new_tokens=decode_kwargs.get("max_new_tokens", 24),
        num_beams=decode_kwargs.get("num_beams", 3),
        temperature=decode_kwargs.get("temperature", 1.0),
        top_p=decode_kwargs.get("top_p", 1.0),
        no_repeat_ngram_size=decode_kwargs.get("no_repeat_ngram_size", 3),
        repetition_penalty=decode_kwargs.get("repetition_penalty", 1.1),
    )
    # 你的 decoder.generate 返回 list[str] 或 str，两种都兜底
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    return clean_text(text)

# ------------------------- 对外可复用：单视频推理接口 -------------------------
@torch.no_grad()
def run_one_video(
    frames_dir: str,
    ckpt: str,
    stage: str = "all",
    *,
    vit_name: str = "vit_base_patch16_224",
    gpt2_name: str = "gpt2",
    prefix_len: int = 4,
    num_frames: int = 8,
    image_size: int = 224,
    ln_scale: float = 0.6,
    in_weight: float = 0.4,
    preset1: str = "precise",
    preset2: str = "precise",
    preset3: str = "natural",
    prompt1: str = "",
    prompt2: str = "State the main action in one short sentence:",
    prompt3: str = "Write a short, natural caption:",
    emit_json: bool = False,
    **kwargs,  # 容忍多余参数，避免今后再炸
):
    """
    单个视频（帧目录）三路生成 + 选优。
    返回 dict: {"S1":..., "S2":..., "S3":..., "BEST": {"key": "...", "text": "..."}}
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 读帧
    frames = load_frames(Path(frames_dir), num_frames=num_frames, image_size=image_size, device=device)

    # 构建模型（接收 vit_name/gpt2_name/prefix_len）
    model = VideoCaptionModel(
        vit_name=vit_name,
        gpt2_name=gpt2_name,
        cond_mode="prefix",
        prefix_len=prefix_len,
        freeze_vit=True,
        unfreeze_last=0,
    ).to(device).eval()

    # 加载 ckpt（兼容 state['model_state'] 或直接 state_dict）
    ckpt_path = Path(ckpt)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    logging.getLogger(__name__).info(f"[ckpt] keys={len(state.keys())}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logging.getLogger(__name__).warning(f"missing keys (<=6): {missing[:6]}")
    if unexpected:
        logging.getLogger(__name__).warning(f"unexpected keys (<=6): {unexpected[:6]}")

    # 三路生成
    s1 = generate_once(model, frames, prompt1, ln_scale, in_weight, **preset_to_kwargs(preset1))
    s2 = generate_once(model, frames, prompt2, ln_scale, in_weight, **preset_to_kwargs(preset2))
    s3 = generate_once(model, frames, prompt3, ln_scale, in_weight, **preset_to_kwargs(preset3))

    # 选优
    scored = [(k, v, _score_sentence(v)) for k, v in [("S1", s1), ("S2", s2), ("S3", s3)]]
    best_key, best_text, _ = sorted(scored, key=lambda x: x[2], reverse=True)[0]

    result = {"S1": s1, "S2": s2, "S3": s3, "BEST": {"key": best_key, "text": best_text}}

    # 供命令行模式用；FastAPI 里我们直接返回 dict
    if emit_json:
        print(json.dumps(result, ensure_ascii=False))

    return result

# ------------------------- 主流程 -------------------------
def parse_args():
    p = argparse.ArgumentParser("Run caption generation on a frames directory")
    p.add_argument("--frames_dir", required=True, help="目录下需存在 frame_*.jpg")
    p.add_argument("--stage", choices=["1","2","3","all"], default="all")
    p.add_argument("--ckpt", required=True, help="*.pt，需含 decoder.* / mapper.* 等生成相关权重")
    p.add_argument("--vit_name", default="vit_base_patch16_224")
    p.add_argument("--gpt2_name", default="gpt2")
    p.add_argument("--prefix_len", type=int, default=4)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--ln_scale", type=float, default=0.6)
    p.add_argument("--in_weight", type=float, default=0.4)
    p.add_argument("--preset1", choices=["precise","detailed","natural","safe_sample"], default="precise")
    p.add_argument("--preset2", choices=["precise","detailed","natural","safe_sample"], default="precise")
    p.add_argument("--preset3", choices=["precise","detailed","natural","safe_sample"], default="natural")
    p.add_argument("--prompt1", default="")
    p.add_argument("--prompt2", default="State the main action in one short sentence:")
    p.add_argument("--prompt3", default="Write a short, natural caption:")
    p.add_argument("--emit_json", action="store_true")
    p.add_argument("--log_level", default="INFO")
    return p.parse_args()




def main():
    args = parse_args()
    setup_logging(args.log_level)

    out = run_one_video(
        frames_dir=args.frames_dir,
        ckpt=args.ckpt,
        vit_name=args.vit_name,
        gpt2_name=args.gpt2_name,
        prefix_len=args.prefix_len,
        num_frames=args.num_frames,
        image_size=args.image_size,
        ln_scale=args.ln_scale,
        in_weight=args.in_weight,
        presets=(args.preset1, args.preset2, args.preset3),
        prompts=(args.prompt1, args.prompt2, args.prompt3),
        emit_json=args.emit_json,
    )

    if not args.emit_json:
        print(f"[BEST] {out['BEST']['key']}: {out['BEST']['text']}")
        print(f"[S1] {out['S1']}")
        print(f"[S2] {out['S2']}")
        print(f"[S3] {out['S3']}")

if __name__ == "__main__":
    main()