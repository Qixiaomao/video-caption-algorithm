# src/models/vit_text_align.py
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None

class ViTTextAlignModel(nn.Module):
    """
    使用说明：输入尺寸 [B,T,3,224,224] 视频，[B,L] 文本，输出为损失
    默认时间平均：(simple & input)
    视频侧：逐帧用 ViT 提特征 -> [B,T,C] 时间平均 -> [B,C] -> proj -> L2
    文本侧：Embedding -> TransformerEncoder(可选层) -> mask 平均 -> proj -> L2
    损失：CosineEmbeddingLoss
    """
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        vit_name: str = "vit_base_patch16_224",
        txt_dim: int = 512,
        vid_dim: int = 768,   # 对应 vit_base 的输出维度
        proj_dim: int = 256,
        txt_layers: int = 2,
        txt_nhead: int = 8,
        dropout: float = 0.1,
        freeze_vit: bool = False,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("Please `pip install timm` for ViT backbones.")

        # ---- Video encoder (frame-wise ViT) ----
        self.vit = timm.create_model(vit_name, pretrained=True, num_classes=0)  # 输出 [B, vid_dim]
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False
        self.vid_proj = nn.Linear(vid_dim, proj_dim)

        # ---- Text encoder ----
        self.pad_id = pad_id
        self.txt_emb  = nn.Embedding(vocab_size, txt_dim, padding_idx=pad_id)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=txt_dim, nhead=txt_nhead, dim_feedforward=txt_dim*4, dropout=dropout, batch_first=True
        )
        self.txt_enc  = nn.TransformerEncoder(encoder_layer, num_layers=txt_layers) if txt_layers>0 else nn.Identity()
        self.txt_proj = nn.Linear(txt_dim, proj_dim)

        # ---- Loss ----
        self.loss_fn = nn.CosineEmbeddingLoss()

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        """
        video: [B,T,3,224,224]
        return: [B,proj_dim] (L2 normalized)
        """
        B, T, C, H, W = video.shape
        x = video.reshape(B*T, C, H, W)              # 合并帧维
        feat = self.vit(x)                           # [B*T, vid_dim]
        feat = feat.reshape(B, T, -1).mean(dim=1)    # 时间平均 [B, vid_dim]
        feat = self.vid_proj(feat)                   # [B, proj_dim]
        feat = nn.functional.normalize(feat, dim=-1)
        return feat

    def encode_text(self, caption_ids: torch.Tensor) -> torch.Tensor:
        """
        caption_ids: [B,L]
        return: [B,proj_dim] (L2 normalized)
        """
        mask = (caption_ids != self.pad_id)          # True for valid tokens
        x = self.txt_emb(caption_ids)                # [B,L,txt_dim]
        x = self.txt_enc(x)                          # [B,L,txt_dim]
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / denom  # mask 平均池化 -> [B,txt_dim]
        x = self.txt_proj(x)                         # [B,proj_dim]
        x = nn.functional.normalize(x, dim=-1)
        return x

    def forward(self, video: torch.Tensor, caption_ids: torch.Tensor) -> torch.Tensor:
        v = self.encode_video(video)
        t = self.encode_text(caption_ids)
        target = torch.ones(video.size(0), device=video.device)
        loss = self.loss_fn(v, t, target)
        return loss
