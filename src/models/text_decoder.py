# src/models/text_decoder.py
# -*- coding: utf-8 -*-
from typing import Optional, List
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

class GPT2TextDecoder(nn.Module):
    """
    将视频向量映射到 GPT-2 的条件输入并生成文本。
    不要在本文件 import caption_model，避免循环依赖。
    """
    def __init__(self,
                 gpt2_name: str = "gpt2",
                 cond_mode: str = "prefix",   # 'prefix' or 'bos'
                 prefix_len: int = 4,
                 video_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(gpt2_name)
        self.model = GPT2LMHeadModel.from_pretrained(gpt2_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.cond_mode = cond_mode
        self.prefix_len = prefix_len
        hid = self.model.config.n_embd

        if cond_mode == "prefix":
            self.mapper = nn.Sequential(
                nn.Linear(video_dim, hid * prefix_len),
                nn.Dropout(dropout)
            )
        elif cond_mode == "bos":
            self.mapper = nn.Sequential(
                nn.Linear(video_dim, hid),
                nn.Tanh(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError("cond_mode must be 'prefix' or 'bos'")

    # ---- 内部工具：根据 cond_mode 组装 inputs_embeds ----
    def _build_inputs(self, video_emb: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        B = video_emb.size(0)
        hid = self.model.config.n_embd
        base_embeds = self.model.transformer.wte(input_ids)      # [B,L,H]
        if base_embeds.size(0) == 1 and B > 1:
            base_embeds = base_embeds.expand(B, -1, -1)

        if self.cond_mode == "prefix":
            mapped = self.mapper(video_emb).view(B, self.prefix_len, hid)  # [B,P,H]
            inputs_embeds = torch.cat([mapped, base_embeds], dim=1)        # [B,P+L,H]
        else:  # 'bos'
            bos_embed = self.mapper(video_emb).unsqueeze(1)                 # [B,1,H]
            inputs_embeds = torch.cat([bos_embed, base_embeds], dim=1)      # [B,1+L,H]
        return inputs_embeds

    def forward(self,
                video_emb: torch.Tensor,           # [B, D]
                input_ids: torch.Tensor,           # [B, L]
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        B = video_emb.size(0)

        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 为 prefix/bos 补 mask
        extra = self.prefix_len if self.cond_mode == "prefix" else 1
        extra_mask = torch.ones(B, extra, dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([extra_mask, attention_mask], dim=1)

        inputs_embeds = self._build_inputs(video_emb, input_ids)

        loss = None
        if labels is not None:
            pad = torch.full((B, extra), -100, dtype=labels.dtype, device=labels.device)
            ext_labels = torch.cat([pad, labels], dim=1)
            out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=ext_labels)
            loss = out.loss
        else:
            out = self.model(inputs_embeds=inputs_embeds, attention_mask=attn)

        return {"loss": loss, "logits": out.logits}

    @torch.no_grad()
    def generate(self,
                 video_emb: torch.Tensor,
                 prompt: str = "",
                 max_new_tokens: int = 32,
                 num_beams: int = 1,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 no_repeat_ngram_size: int = 3,
                 repetition_penalty: float = 1.15,
                 min_new_tokens: int = 8) -> List[str]:
        device = video_emb.device
        if prompt:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        else:
            input_ids = torch.tensor([[self.tokenizer.bos_token_id]], device=device)

        inputs_embeds = self._build_inputs(video_emb, input_ids)

        out_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            do_sample=(num_beams == 1 and temperature != 1.0),
            temperature=temperature,
            top_p=top_p,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        texts = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]