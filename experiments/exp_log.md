## 18/09/25 Experiment log

## 下载数据集
- 处理数据集 kaggle msvd 
- 清洗字幕部分数据
- 按video_id 批量从YouTube拉原视频并裁剪片段

### 终端记录
d_prepare.py --raw_dir .\data\raw\msvd\ --out_dir .\data\processed\msvd --format grouped       
[ERROR] 未能从 data\raw\msvd\annotations.txt 解析出任何 (video_id, caption)python .\scripts\msvd_prepare.py --raw_dir .\data\raw\msvd\ --out_dir .\data\processed\msvd --format groupedS D:\Mylearn\INTI courses\Graduation thesis\video-captioning-transformer>
[INFO] loaded caption rows: 80827
[INFO] unique videos in annotations: 1970
[INFO] indexed video files: 0
[OK] train: 1576 -> data\processed\msvd\train\annotations.json
[OK] val: 197 -> data\processed\msvd\val\annotations.json
[OK] test: 197 -> data\processed\msvd\test\annotations.json
PS D:\Mylearn\INTI courses\Graduation thesis\video-captioning-transformer> python .\check_videos.py
总共找到视频文件: 0
前20个文件示例：


## 对每帧视频进行预处理，加载器读取小样本视频，并裁剪成固定大小的片段

PS D:\Mylearn\INTI courses\Graduation thesis\video-captioning-transformer> python .\scripts\msvd_compat_frame_names_plus.py
[train] 兼容命名完成的目录数: 260
[val] 兼容命名完成的目录数: 0
[test] 兼容命名完成的目录数: 0
[DONE] 兼容命名 V2 完成
PS D:\Mylearn\INTI courses\Graduation thesis\video-captioning-transformer> python -m src.test_loader
[INFO] Using HuggingFace BertTokenizerFast.
[INFO] build_dataloader signature: (ann_path: str, tokenizer, batch_size: int = 2, max_len: int = 32, num_frame: int = 8, image_size: int = 224, shuffle: bool = False, num_wokers: int = 0)  
[INFO] DataLoader created. Iterate a few batches...
---- Batch 0 ----
video: <class 'torch.Tensor'> torch.Size([2, 8, 3, 224, 224])
caption_ids: <class 'torch.Tensor'> torch.Size([2, 32])
video_id: <class 'list'> None
---- Batch 1 ----
video: <class 'torch.Tensor'> torch.Size([2, 8, 3, 224, 224])
caption_ids: <class 'torch.Tensor'> torch.Size([2, 32])
video_id: <class 'list'> None
---- Batch 2 ----
video: <class 'torch.Tensor'> torch.Size([2, 8, 3, 224, 224])
caption_ids: <class 'torch.Tensor'> torch.Size([2, 32])
video_id: <class 'list'> None
[DONE] loader smoke test finished.

## Step 6 | 训练模型

📌 今天的遇到的问题

数据集路径不一致

annotations.json 里 frames_dir 字段带有 -videoID_start_end 格式。

但实际抽帧的目录名多是 videoID 或 videoID_...，导致对不上。

过滤脚本 kept: 0 的问题

原因一：frames_dir 路径错误，没有对齐。

原因二：计数函数用 glob(as_posix()) 在 Windows 下失效，导致即便目录有图，也统计为 0。

解决：换用 pathlib.iterdir() 重新实现计数，确认能返回正确帧数。

训练时报错 No frames found

因为 DataLoader 里写死只匹配 frame_*.jpg。

解决：改成更通用的 loader → 支持 *.jpg/*.png 等后缀，并允许递归查找。

captions 使用问题

原代码只取第一条 caption。

已改进：随机选一条 caption，增强训练多样性。

今天的进度

环境准备

base.yaml + .env 已完成，基本依赖环境准备好。

数据处理

抽帧验证成功，可以生成 jpg 帧。

补丁脚本 + 过滤脚本完成，已成功修复一部分样本路径。

train/annotations.filtered.json 生成，并确认有 kept: 97 可用样本。

训练流程

修改 DataLoader → 兼容多格式帧文件。

成功跑通 20 个 batch 的 smoke test：
```python:
loss=6.9, steps=20
checkpoint -> checkpoints/msvd_debug/simple_vc_smoke.pt
```
- pipeline(数据加载->模型前向/反向传播->参数更新->checkpoint保存) 闭环打通。

#### 当前进度定位

已完成：小规模实验闭环跑通（smoke test）。

待推进：

扩大过滤后的样本量（修补更多 frames_dir，目标几百~几千条）。

提高 --min_frames 回到 8，保证帧质量。

扩大训练规模（更多 batch/epoch），观察 loss 曲线。

准备 inference 测试，验证 caption 输出效果。

## Step 7 

```c:
>>> print("train annotations:",ann.as_posix())
train annotations: data/processed/msvd/train/annotations.json
>>> print("total:",len(recs),"with_frames:",ok)
total: 203 with_frames: 0
>>>

```
短跑了2~3个batch,正常输出。
```c:
PS D:\Mylearn\INTI courses\Graduation thesis\video-captioning-transformer> python -m src.cli.train --ann_path data/processed/msvd/train/annotations.json --batch_size 2 --num_frame 8 --image_size 224 --max_len 32 --seed 123
[DEBUG] ann_path used by train: data/processed/msvd/train/annotations.json
[WARN] Dropped 176 samples without frames. kept=27
---- Batch 0 ----
video: torch.Size([2, 8, 3, 224, 224])
caption_ids: torch.Size([2, 32])
video_id: ['5JSbxHECb-I_97_110', '8PQiaurIiDM_94_99']
---- Batch 1 ----
video: torch.Size([2, 8, 3, 224, 224])
caption_ids: torch.Size([2, 32])
video_id: ['45AGQSbodbU_5_15', '1dfR0A_BXjw_590_600']
---- Batch 2 ----
video: torch.Size([2, 8, 3, 224, 224])
caption_ids: torch.Size([2, 32])
video_id: ['-pUwIypksfE_13_23', '8HB7ywgJuTg_131_142']

```

提升覆盖率(需要提升到80%以上，保证训练量的稳定性)
从 13.3% 到样本量的 
[DONE] split=train ok=169 fail=7
[COVERAGE] train: with_frames=196/203 (96.6%)

可以跑全量的数据指令
```c:
# 只补缺，4线程，2FPS
python -m scripts.extract_frames_mp --splits train --fps 2 --workers 4 --only-missing

# 如果想全量重抽
python -m scripts.extract_frames_mp --splits train --fps 2 --workers 4 --overwrite

```
### 训练验证环节

```c:

PS D:\Mylearn\INTI courses\Graduation thesis\video-captioning-transformer> python -m src.cli.train --ann_path data/processed/msvd/train/annotations.json --batch_size 2 --num_frame 8 --image_size 224 --max_len 32 --seed 123 --epochs 1 --max_steps 50 --lr 5e-4
[DEBUG] ann_path used by train: data/processed/msvd/train/annotations.json
[WARN] Dropped 7 samples without frames. kept=196
step 0001 | loss 1.0457
step 0010 | loss 0.9351
step 0020 | loss 0.6001
step 0030 | loss 0.5465
step 0040 | loss 0.4904
step 0050 | loss 0.7053

```