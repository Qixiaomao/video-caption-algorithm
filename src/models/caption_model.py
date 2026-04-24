# src/models/caption_model.py
# -*- coding: utf-8 -*-
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch, torch.nn.functional as F

from src.models.video_encoder import build_vit_encoder
from src.models.text_decoder import GPT2TextDecoder

class VideoCaptionModel(nn.Module):
    # INFO(core-model): this is the in-house captioning model loaded by the
    # resident inference path in the current startup flow.
    # TODO(decouple): expose this through a stable runtime/service interface.
    """
    闁荤喐鐟ュΛ婵嬨€傞崜浣规殰?-> ViT 缂傚倸鍊归悧婊堟偉?-> (闂佸憡鐟崹鍫曞焵?MLP) -> GPT-2 闂佸搫顦埀顒€寮堕浠嬫煟閵忋垹鏋戦柛?
    闂佸憡鐟禍娆戞妞嬪孩灏?text_decoder闂佹寧绋戞總鏃傜箔婢跺本鍟哄ù锝嗙摃閸╁矂鏌熺€涙ê濮囩€殿喕绮欏畷?import闂?
    """
    def __init__(self,
                 vit_name: str = "vit_base_patch16_224",
                 video_dim: int = 256,
                 gpt2_name: str = "gpt2",
                 cond_mode: str = "prefix",
                 prefix_len: int = 4,
                 proj_hidden: int = 0,
                 freeze_vit: bool = True,
                 unfreeze_last: int = 0,
                 vit_enable_fp16: bool = False,
                 vit_enable_attention_fastpath: bool = True,
                 vit_prefer_channels_last: bool = True,
                 vit_enable_torch_compile: bool = True,
                 vit_torch_compile_mode: str = "reduce-overhead",
                 vit_enable_mlp_bias_gelu_fusion: bool = True,
                 vit_enable_residual_layernorm_fusion: bool = True,
                 vit_enable_inplace_residual_add_fusion: bool = True,
                 vit_enable_cupy_fused_pool: bool = False,
                 vit_cupy_pool_force_fp16: bool = True,
                 use_cupy_prefix_projector: bool = False,
                 cupy_prefix_force_fp16: bool = True):
        super().__init__()
        self.encoder = build_vit_encoder(
            model_name=vit_name,
            out_dim=video_dim,
            pretrained=True,
            pool="cls",
            l2norm=False,
            freeze=freeze_vit,
            unfreeze_last=unfreeze_last,
            enable_attention_fastpath=vit_enable_attention_fastpath,
            prefer_channels_last=vit_prefer_channels_last,
            enable_torch_compile=vit_enable_torch_compile and freeze_vit,
            torch_compile_mode=vit_torch_compile_mode,
            enable_amp_fp16=vit_enable_fp16,
            enable_mlp_bias_gelu_fusion=vit_enable_mlp_bias_gelu_fusion,
            enable_residual_layernorm_fusion=vit_enable_residual_layernorm_fusion,
            enable_inplace_residual_add_fusion=vit_enable_inplace_residual_add_fusion,
            enable_cupy_fused_pool=vit_enable_cupy_fused_pool,
            cupy_pool_force_fp16=vit_cupy_pool_force_fp16,
        )
        if proj_hidden > 0:
            self.proj = nn.Sequential(
                nn.Linear(video_dim, proj_hidden),
                nn.ReLU(),
                nn.Linear(proj_hidden, video_dim),
            )
        else:
            self.proj = nn.Identity()

        self.decoder = GPT2TextDecoder(
            gpt2_name=gpt2_name,
            cond_mode=cond_mode,
            prefix_len=prefix_len,
            video_dim=video_dim,
            use_cupy_prefix_projector=use_cupy_prefix_projector,
            cupy_prefix_force_fp16=cupy_prefix_force_fp16,
        )
        # INFO(core-ops):
        # - encoder: vision feature extraction hot path
        # - proj: visual adaptation/projection point
        # - decoder.mapper: visual-to-language bridge
        # - decoder.model: GPT-2 decode hot path

    def forward(self,
                video: torch.Tensor,                # [B,T,3,224,224]
                input_ids: torch.Tensor,            # [B,L]
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        video_emb = self.encoder(video)             # [B, D]
        video_emb = self.proj(video_emb)            # [B, D]
        out = self.decoder(video_emb, input_ids, attention_mask, labels)
        return out

    @torch.no_grad()
    def generate(self,
                 video: torch.Tensor,
                 prompt: str = "",
                 **gen_kwargs):
        # INFO(core-ops): most relevant runtime path for latency profiling.
        video_emb = self.encoder(video)
        video_emb = self.proj(video_emb)
        return self.decoder.generate(video_emb, prompt=prompt, **gen_kwargs)
    
    # --- 闂?VideoCaptionModel 闂佸憡鍔曢幊姗€宕曢幘顔兼闁哄顑欓弶濠氭煥濞戞澧涢柛銊ｅ妽閹峰懎顭ㄩ埀顒傛嫻閻旂厧瑙﹂悘鐐靛亾閻庮喖霉閻樻彃顣虫繛鍫熷灦濞煎繘宕ㄩ鐐愭﹢鏌涢幋锝呅撻柡鍡欏枑缁嬪﹪鎮㈠ú缁樻杸闂?--
    def compute_loss(self, video: torch.Tensor, caption_ids: torch.Tensor, pad_id: int):
        """
        闁荤姳绶ょ槐鏇㈡偩閼姐倖鍠嗛柛鈩冧緱閺?闂佸搫鍊稿ú锕€锕㈤幍顔瑰亾闂堟稒璐℃繛鍫熷灦缁傚秹濡堕崨顓犻獓闂佺粯鏌ㄩ悘姘暦椤栨稑绶炲〒姘氨閸?
        video: [B,T,3,H,W]
        caption_ids: [B,L]闂佹寧绋戦懟顖溾偓鍨耿瀹?<bos> 闂?<eos>闂佹寧绋戦濉ding 闂?pad_id
        闁哄鏅滈弻銊ッ? 闂佸搫绉村ú顓㈠闯濞差亜绠查柣鏃€鐏氭禍?
        """
        device = video.device
        # 1) 闁荤喐鐟ュΛ婊堬綖鎼淬劌绀堢€广儱娲ㄧ壕濠氭煥濞戞澧曢柟?generate() 闂佸憡鑹鹃張顒勵敆閻愮儤鏅?
        video_emb = self.encoder(video)     # 閻熸粏鍩囬崹缁樻叏閸涘﹦鐟规繝闈涙噽缁嬫垿鏌?encoder
        prefix    = self.proj(video_emb)    # 闂佸憡鐟崹鐢稿礂濮椻偓瀵?[B, Dp] 闂?[B, P, Dp]
        if prefix.dim() == 2:
            prefix = prefix.unsqueeze(1)    # -> [B, 1, Dp]

        # 2) GPT-2 闂備焦婢樼粔鍫曟偪?
        gpt2 = getattr(self.decoder, "model", self.decoder)   # 闂佺绻掗崢褔顢?wrapper
        hidden = gpt2.config.n_embd                           # 闂備緡鍋呴懝楣冩偉?768
        B, P, Dp = prefix.shape

        # 3) 闁?prefix 闂佸搫鐗冮崑鎾绘煕濮橆剛鍑圭紒鏂款嚟缁辨帡寮堕幋顓炴婵帞绮崝鏇㈠春?GPT-2 闂傚倸鎳忛崝妯何涘畝鈧槐?
        if Dp != hidden:
            # 3a) 婵炴潙鍚嬮敋闁告ɑ绋掗幏鍛崉閵婏附娈?decoder.mapper闂佹寧绋戦悧鎰偓姘ュ灮閳ь剚绋掗敋婵犫偓椤忓牊鏅?
            if hasattr(self.decoder, "mapper") and self.decoder.mapper is not None:
                # 闁汇埄鍨遍悺鏇綖?mapper 闂佽　鍋撴い鏍ㄧ☉閻?[B,P,Dp] 闂?[B,Dp]
                try:
                    prefix = self.decoder.mapper(prefix)      # 闂佸搫鐗忛崰鎰耿?-> [B,P,hidden]
                except Exception:
                    # 闁诲繐绻戠换鍡涙儊椤栫偛绀傞柛顐ｇ箖閸娿倖顨?P 缂?
                    prefix_flat = prefix.reshape(B, P*Dp)
                    prefix = self.decoder.mapper(prefix_flat).reshape(B, -1, hidden)
            # 3b) 婵犵鈧啿鈧綊鎮?proj 闁哄鐗婇幐鎼佸吹椤撶儐鍟呴柤纰卞墰閻ュ懘鏌?[B, P*hidden]闂佹寧绋戦懟顖炲垂椤栫偞鍎庨悗娑櫭径?reshape
            if prefix.shape[-1] != hidden:
                if prefix.shape[-1] % hidden == 0:
                    P2 = prefix.shape[-1] // hidden
                    prefix = prefix.reshape(B, P*P2, hidden)
                    P = prefix.shape[1]
                else:
                    raise RuntimeError(
                        f"[compute_loss] prefix dim mismatch: got Dp={Dp}, but GPT2 hidden={hidden}. "
                        "Set self.proj out_features to hidden * prefix_len (e.g., 768 * P), "
                        "or provide decoder.mapper to map Dp to hidden."
                    )

        # 4) teacher-forcing闂佹寧绋掓穱铏规椤撱垹绀?闂佸搫绉村ú銊╊敆閻戣姤鐓ユ繛鍡楁捣椤忓崬霉?
        inp_ids = caption_ids[:, :-1].contiguous()  # [B,L-1]
        labels  = caption_ids[:, 1:].contiguous()   # [B,L-1]

        # 5) 闂佺懓鍢查崥瀣暜鐎涙ɑ缍囬柟鎯у暱瀵磭鈧鎸搁懟顖炲矗閸℃稒鏅慨婵堟箷efix + token embedding
        wte  = gpt2.transformer.wte                 # 闁荤姴娲ょ粔瀵哥矈閻㈢绀傞柕澶堝劤濠€?
        tok_emb = wte(inp_ids)                      # [B,L-1,hidden]
        inputs_embeds = torch.cat([prefix, tok_emb], dim=1)  # [B,P+L-1,hidden]

        # 6) attention mask / labels 闁诲酣娼х紞濠勭礊?
        att_txt   = (inp_ids != pad_id).long()                   # [B,L-1]
        attn_mask = torch.cat([torch.ones((B, P), device=device, dtype=torch.long),
                            att_txt], dim=1)                  # [B,P+L-1]
        ignore    = -100
        pad_prefix = torch.full((B, P), ignore, dtype=torch.long, device=device)
        lm_labels  = torch.cat([pad_prefix, labels], dim=1)      # [B,P+L-1]

        out = gpt2(inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                labels=lm_labels,
                use_cache=False)
        return out.loss

    










