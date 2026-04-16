#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, logging, os, re, sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# 浣犻」鐩唴鐨勬ā鍨嬬被
from src.models.caption_model import VideoCaptionModel

# LEGACY(experiment): this module is kept for standalone experiments and
# experiments/hybrid_infer.py compatibility. Product inference now flows
# through server.app -> server.services -> core.engine -> core.models.


# ------------------------- 鍩虹璁炬柦 -------------------------
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


# ------------------------- 鏂囨湰娓呮礂宸ュ叿 -------------------------
def _strip_acronyms_and_countries(s: str) -> str:
    s = re.sub(r"\bU\.S\.A?\.?\b", "", s, flags=re.I)
    s = re.sub(r"\bUSA\b", "", s, flags=re.I)
    s = re.sub(r"\bUnited States of America\b", "", s, flags=re.I)
    s = re.sub(r"\bUnited States\b", "", s, flags=re.I)
    s = re.sub(r"\bAmerica\b", "", s, flags=re.I)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _collapse_prep_chain(s: str) -> str:
    # 鍘嬬缉浠嬭瘝閾撅細in the front of / in front of / in the middle of ... 绛?
    s = re.sub(r"(?i)\bin\s+the\s+front\s+of\b", "in front of", s)
    s = re.sub(r"(?i)\bin\s+the\s+middle\s+of\b", "in the middle of", s)
    s = re.sub(r"(?i)\bat\s+the\s+side\s+of\b", "at the side of", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s


def _ensure_sit_complement(s: str) -> str:
    """
    鑻ュ彞瀛愬彧鏈?'Someone is sitting' 鎴栧潗濮跨己灏戝湴鐐?瀹捐锛岃ˉ涓€涓俯鍜屽畨鍏ㄧ殑琛ヨ銆?
    閬囧埌宸茬粡鏄?'Someone is ...' 鐨勫畬鏁村彞瀛愬垯涓嶅啀鎷兼帴锛岄伩鍏嶉噸澶嶃€?
    """
    low = s.strip().lower()

    # 宸叉湁鑷劧鐨?'someone is ...'锛屼笉鍐嶆敼鍐?
    if re.match(r"^someone\s+is\b", low):
        return s

    # 鍙湁 'someone is sitting'锛堟棤琛ヨ锛?
    if re.match(r"^someone\s+is\s+sitting\s*\.?$", low):
        return "Someone is sitting on a chair."

    # sitting 寮€澶翠絾娌′粙璇嶈ˉ璇紝鎷间竴涓?
    if re.match(r"^someone\s+is\s+sitting\b", low) and not re.search(r"\b(in|on|at|by|with|near)\b", low):
        return s.rstrip(". ") + " on a chair."
    return s


def _truncate_on_noise(s: str) -> str:
    """
    涓€鏃﹀彞瀛愰噷鍑虹幇鏄庢樉鍣０ token锛堝惈鏁板瓧/鏂滄潬/澶у啓缂╁啓/杩炲瓧绗︾缉鍐欑瓑锛夛紝
    绔嬪埢浠庤 token 涔嬪墠鎴柇锛屼繚璇佺粨灏炬湁鍙ュ彿銆?
    """
    if not s:
        return s
    toks = s.split()
    cut = len(toks)

    for i, t in enumerate(toks):
        t_raw = t.strip(",.;:!?()[]{}\"'`")
        if not t_raw:
            continue
        if re.search(r"[0-9/\\]", t_raw):                 # 鍚暟瀛?鏂滄潬
            cut = i; break
        if re.match(r"^(?:[A-Za-z]\.){2,}$", t_raw):      # A.B. / I.D. / U.S.
            cut = i; break
        if re.match(r"^[A-Z]{1,3}-[A-Za-z0-9]{1,6}$", t_raw):  # W-8 / 1099-MISC
            cut = i; break
        if len(t_raw) <= 3 and t_raw.isupper():           # 瀛ょ珛澶у啓缂╁啓
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
    # 杩炵画閲嶅璇嶅幓閲嶏紙娉ㄦ剰 raw \1锛?
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
    
    # 鈥斺€?寮轰竴鐐圭殑缃戦〉鑵斾笌鐮存姌鍙锋竻鐞嗭紙鍙共鎺夊櫔澹帮紝涓嶇姝ｅ父鍙ワ級鈥斺€?
    # 0) 鍏ㄦ槸鐮存姌鍙?涓嬪垝绾?瑁呴グ绾匡細缃┖
    if re.fullmatch(r'[-鈥擾=\s]{6,}\.?', s):
        return ""

    # 1) 琛岄鐮存姌鍙?涓嬪垝绾?+ 绌烘牸锛氬幓鎺夛紙寰堝缃戦〉妯℃澘鍠滄杩欐牱寮€澶达級
    s = re.sub(r'^\s*[-鈥擾=\s]{2,}\s*', "", s)

    # 2) 绾摼鎺?HTML/鐗堟潈/寮曞彿楦℃堡锛氱洿鎺ョ疆绌?
    if re.match(r'^\s*(https?://|www\.|<a\b|&lt;a\b)', s, re.I) \
    or re.match(r'^\s*(漏|copyright\b)', s, re.I) \
    or re.fullmatch(r'"\s*[^"]+\s*"\.?', s):
        return ""

    # 3) 甯歌鈥滆惀閿€/鍗冲皢鎾斁/鐐瑰嚮鈥濇ā鏉垮彞瑙﹀彂锛氱疆绌?
    BAD_LEADS = (
        r"you are about to\b", r"click here\b", r"subscribe\b",
        r"available on youtube\b", r"watch live\b", r"find out\b",
        r"the video will\b", r"on the road\b"
    )
    if re.match(r'^\s*(?:' + "|".join(BAD_LEADS) + r')', s, re.I):
        return ""

    # 4) 濡傛灉鍖呭惈鏄庢樉 HTML 娈嬬墖/绀句氦绔欏悕锛氱疆绌?
    if re.search(r'(</?\w+>|reddit\.com|pastebin|mailto:)', s, re.I):
        return ""
        
    # 鈥斺€?缁嗗寲娓呯悊姝ラ 鈥斺€?

    # 0) 蹇€熻繃婊ょ綉椤佃厰 & 璁板綍鏍囪
    flagged = bool(re.search(r"(?i)\b(click here|subscribe|report abuse|pastebin|official facebook|video will be)\b", s))
    s = re.sub(r"(?i)\b(click here|subscribe|report abuse|pastebin|official facebook|video will be.*)$", "", s).strip()

    # 1) 鍘诲浗瀹剁缉鍐?缃戦〉灏惧反
    s = _strip_acronyms_and_countries(s)

    # 2) 浠嬭瘝閾惧帇缂?
    s = _collapse_prep_chain(s)

    # 3) 閬囧埌鍣０ token 鐩存帴鎴柇
    if len(s.split()) >= 10:
        s = _truncate_on_noise(s)

    # 4) 鍓帀濂囨€熬宸?
    s = _prune_weird_tails(s)

    # 5) 鑻ョ綉椤佃厰娓呯┖/杩囩煭锛岀粰涓畨鍏ㄥ厹搴曪紙鍦ㄥ潗濮胯ˉ璇墠鍚庨兘鍙紝杩欓噷鏀惧墠锛?
    if flagged and len(s.split()) <= 2:
        s = "Someone is in the scene."

    # 6) 鍧愬Э琛ヨ锛堜粎鍦ㄥ彞寮忛渶瑕佹椂瑙﹀彂锛?
    s = _ensure_sit_complement(s)

    # 7) 鍘婚噸銆侀瀛楁瘝/鍙ユ湯鏍囩偣
    s = _dedup_tokens(s)
    s = _ensure_period_and_caps(s)

    # 8) 鑻ュ鍙ワ紝閫夊垎鏈€楂樼殑涓€鍙?
    parts = re.split(r"\s*(?<=\.|\!|\?)\s+", s)
    parts = [t.strip() for t in parts if t.strip()]
    if len(parts) > 1:
        s = max(parts, key=_score_sentence)

    # 9) 鍏滃簳锛氱涓€鍙?
    s = _first_sentence(s)
    return s


# ------------------------- 鐢熸垚 -------------------------
def preset_to_kwargs(name: str):
    name = (name or "precise").lower()
    if name == "precise":
        return dict(num_beams=3, max_new_tokens=24, temperature=1.0,
                    top_p=1.0, no_repeat_ngram_size=3, repetition_penalty=1.1)
    if name == "detailed":
        return dict(num_beams=4, max_new_tokens=40, temperature=1.0, # max_new_tokens 鏇村ぇ 32->40
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

    # INFO(core-ops): latency-critical path:
    # ViT encode -> projection/normalization -> GPT-2 generation.
    # TODO(decouple): split preproc / model forward / postprocess for
    # isolated benchmarking and operator-level optimization.

    # 缂栫爜 + prefix 娉ㄥ叆锛堟敮鎸?mapper锛?
    emb = model.encoder(video)            # [B,D] 鎴?[B,*,D]
    emb = model.proj(emb)                 # -> prefix 绌洪棿锛堥€氬父 [B,P,Dp] 鎴?[B,1,Dp]锛?
    # 鑻ユ湁 mapper锛歞ecoder 鍐呴儴浼氬鐞嗭紱鎴戜滑杩欓噷鍙仛杞婚噺褰掍竴鍖栫缉鏀?
    if emb.dim() == 2:
        emb = emb.unsqueeze(1)            # [B,1,Dp]

    # 杞婚噺娉ㄥ叆锛堟寜浣犺缁冩椂鐨勫昂搴︼級
    if ln_scale is not None and ln_scale > 0:
        emb = torch.nn.functional.layer_norm(emb, emb.shape[-1:]) * ln_scale
    if in_weight is not None and in_weight > 0:
        emb = emb * in_weight

    # 鐢熸垚锛堜笉浼?do_sample/logits_processor 閬垮厤鎺ュ彛涓嶅尮閰嶏級
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
    # 浣犵殑 decoder.generate 杩斿洖 list[str] 鎴?str锛屼袱绉嶉兘鍏滃簳
    if isinstance(text, (list, tuple)):
        text = text[0] if text else ""
    return clean_text(text)

# ------------------------- 瀵瑰鍙鐢細鍗曡棰戞帹鐞嗘帴鍙?-------------------------
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
    **kwargs,  # 瀹瑰繊澶氫綑鍙傛暟锛岄伩鍏嶄粖鍚庡啀鐐?
):
    """
    鍗曚釜瑙嗛锛堝抚鐩綍锛変笁璺敓鎴?+ 閫変紭銆?
    杩斿洖 dict: {"S1":..., "S2":..., "S3":..., "BEST": {"key": "...", "text": "..."}}
    """
    # LEGACY(experiment): this path intentionally stays self-contained for
    # old comparisons. Product code should use core.engine.InferenceEngine.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 璇诲抚
    frames = load_frames(Path(frames_dir), num_frames=num_frames, image_size=image_size, device=device)

    # 鏋勫缓妯″瀷锛堟帴鏀?vit_name/gpt2_name/prefix_len锛?
    # INFO(core-model): the in-house runtime model is
    # VideoCaptionModel(ViTFrameEncoder + proj + GPT2TextDecoder).
    # This is the main object to isolate for operator tuning and test coverage.
    model = VideoCaptionModel(
        vit_name=vit_name,
        gpt2_name=gpt2_name,
        cond_mode="prefix",
        prefix_len=prefix_len,
        freeze_vit=True,
        unfreeze_last=0,
    ).to(device).eval()

    # 鍔犺浇 ckpt锛堝吋瀹?state['model_state'] 鎴栫洿鎺?state_dict锛?
    # Product checkpoint IO is handled by core.models.model_loader.
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

    # 涓夎矾鐢熸垚
    s1 = generate_once(model, frames, prompt1, ln_scale, in_weight, **preset_to_kwargs(preset1))
    s2 = generate_once(model, frames, prompt2, ln_scale, in_weight, **preset_to_kwargs(preset2))
    s3 = generate_once(model, frames, prompt3, ln_scale, in_weight, **preset_to_kwargs(preset3))

    # 閫変紭
    scored = [(k, v, _score_sentence(v)) for k, v in [("S1", s1), ("S2", s2), ("S3", s3)]]
    best_key, best_text, _ = sorted(scored, key=lambda x: x[2], reverse=True)[0]

    result = {"S1": s1, "S2": s2, "S3": s3, "BEST": {"key": best_key, "text": best_text}}

    # 渚涘懡浠よ妯″紡鐢紱FastAPI 閲屾垜浠洿鎺ヨ繑鍥?dict
    if emit_json:
        print(json.dumps(result, ensure_ascii=False))

    return result

# ------------------------- 涓绘祦绋?-------------------------
def parse_args():
    p = argparse.ArgumentParser("Run caption generation on a frames directory")
    p.add_argument("--frames_dir", required=True, help="鐩綍涓嬮渶瀛樺湪 frame_*.jpg")
    p.add_argument("--stage", choices=["1","2","3","all"], default="all")
    p.add_argument("--ckpt", required=True, help="*.pt锛岄渶鍚?decoder.* / mapper.* 绛夌敓鎴愮浉鍏虫潈閲?)
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

