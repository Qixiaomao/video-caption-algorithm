'''
day 6
小样本数据集处理->
平均池化特征+线性层 构建小型模型
-> 训练 -> checkpoint

'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

class SimpleVideoCaptioner(nn.Module):
    '''
    一个极简版的视频描述生成模型
    - 输入: [B, T, C, H, W] 的视频张量
    - 输出: [B, seq_len, vocab_size] 的 token logits
    
    '''
    
    def __init__(self, vocab_size=30522,hidden_size=512,max_len=32,pad_id=0):
        super(SimpleVideoCaptioner,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.pad_id = pad_id
        
        # 简单特征压缩: (T,H,W) 平均池化
        # -> [B,C,1,1,1]
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        
        # 投影到隐藏层
        self.fc = nn.Linear(3,hidden_size) # 3 输入层C=3(RGB通道)
        
        # 解码部分：简单的全连接映射到vocab
        self.decoder = nn.Linear(hidden_size,vocab_size)
        
    def forward(self,video, caption=None):
        """
        video: [B, T, C, H, W]
        captions: [B, seq_len] (teacher forcing用, 这里暂时不用)
        """
        
        B, T, C, H, W = video.shape
        
        # 平均池化（时序+空间）
        pooled = video.mean(dim=[1,3,4]) # [B,C]
        
        # 投影到hidden
        h = self.fc(pooled) # [B,hidden_size]
        
        # 扩展成seq_len 步(假装每一步都用同一个视频特征)
        h_expand = h.unsqueeze(1).repeat(1,self.max_len,1) # [B,seq_len,hidden_size]
        
        # 映射到词表大小
        logits = self.decoder(h_expand) # [B,seq_len,vocab_size]
        
        return logits