import torch
import torch.nn as nn

class TinyCaptioner(nn.Module):
    """
    Video-conditioned GRU language model (极简版)
    - 输入: video [B, T, C, H, W], input_ids [B, L]
    - 输出: logits [B, L, V]
    """

    def __init__(self, vocab_size: int, d_model: int = 256, pad_id: int = 0):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.video_proj = nn.Linear(1, d_model)  # 输入视频均值 -> 投影
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, video, input_ids):
        # crude video summary: 平均所有像素 -> [B, T]
        vid_scalar = video.float().mean(dim=(2, 3, 4))      # [B, T]
        vid_scalar = vid_scalar.mean(dim=1, keepdim=True)   # [B, 1]
        vid_scalar = vid_scalar.unsqueeze(-1)               # [B, 1, 1]
        vid_feat = self.video_proj(vid_scalar)              # [B, 1, D]

        # 文本 token embedding
        tok = self.token_emb(input_ids)   # [B, L, D]
        tok = tok + vid_feat              # 视频特征加到所有 token 上

        # GRU 解码
        y, _ = self.gru(tok)              # [B, L, D]
        logits = self.out(y)              # [B, L, V]
        return logits
