from __future__ import annotations

import logging

import torch

from core.config import InferenceConfig
from core.datatypes import CaptionCandidates, InferenceResult
from core.env import assert_core_runtime_ready
from core.inference import preset_to_kwargs
from core.memory import MemoryManager
from core.models.model_loader import load_caption_model
from core.postprocessing.candidate_ranker import select_best
from core.postprocessing.text_cleaner import clean_text
from core.preprocessing.frame_loader import load_video_tensor

log = logging.getLogger(__name__)


class InferenceEngine:
    """Stateless core engine for video caption inference.

    The engine owns model execution and tensor flow only. It does not know
    Chainlit, FastAPI, HTTP requests, sessions, or UI state.
    """

    def __init__(self, config: InferenceConfig):
        # Guard direct core usage as well as server-managed usage.
        assert_core_runtime_ready(device=config.device, require_cupy=config.tensorrt.enabled)
        self.config = config
        self.memory = MemoryManager(config.memory)
        self.device = config.device
        self.model = load_caption_model(config)

    @classmethod
    def from_config(cls, config: InferenceConfig):
        return cls(config)

    @torch.no_grad()
    def _generate_once(self, video: torch.Tensor, prompt: str, **decode_kwargs) -> str:
        # INFO(core-ops): tensor path for future profiling:
        # video -> encoder -> projection -> normalized prefix -> GPT-2 decode.
        emb = self.model.encoder(video)
        emb = self.model.proj(emb)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        if self.config.ln_scale is not None and self.config.ln_scale > 0:
            emb = torch.nn.functional.layer_norm(emb, emb.shape[-1:]) * self.config.ln_scale
        if self.config.in_weight is not None and self.config.in_weight > 0:
            emb = emb * self.config.in_weight

        text = self.model.decoder.generate(
            emb,
            prompt=prompt or "",
            max_new_tokens=decode_kwargs.get("max_new_tokens", 24),
            num_beams=decode_kwargs.get("num_beams", 3),
            temperature=decode_kwargs.get("temperature", 1.0),
            top_p=decode_kwargs.get("top_p", 1.0),
            no_repeat_ngram_size=decode_kwargs.get("no_repeat_ngram_size", 3),
            repetition_penalty=decode_kwargs.get("repetition_penalty", 1.1),
        )
        if isinstance(text, (list, tuple)):
            text = text[0] if text else ""
        return clean_text(text)

    @torch.no_grad()
    def infer(self, frames_dir: str) -> InferenceResult:
        with self.memory.oom_guard():
            video = load_video_tensor(
                frames_dir,
                num_frames=self.config.num_frames,
                image_size=self.config.image_size,
                device=self.device,
            )
            candidates = CaptionCandidates(
                s1=self._generate_once(video, self.config.prompt1, **preset_to_kwargs(self.config.preset1)),
                s2=self._generate_once(video, self.config.prompt2, **preset_to_kwargs(self.config.preset2)),
                s3=self._generate_once(video, self.config.prompt3, **preset_to_kwargs(self.config.preset3)),
            )
            best_key, best_text, _ = select_best(
                [("S1", candidates.s1), ("S2", candidates.s2), ("S3", candidates.s3)]
            )
            return InferenceResult(candidates=candidates, best_key=best_key, best_text=best_text)




