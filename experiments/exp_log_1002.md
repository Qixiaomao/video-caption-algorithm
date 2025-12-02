# 今日实验总结 (2025-10-02)

## 实验内容
检查 val/test split 覆盖率

### 实验步骤

```c:
// 1. 执行抽帧脚本(val/test)
python -m scripts.extract_frames_mp --splits val,test --fps 2 --workers 4 --only-missing

/*
--only-missing 只处理缺帧的样本，已经抽过的不重复做
--fps 2 每秒抽帧(和train保持一致)
--workers 4 并行处理的线程数
*/

// 2.脚本检查覆盖率 ./scripts/test_cov_valtest.py
// 3. Dataloader验证(val/test) 
python -m scripts.check_dataloader --ann_path data/processed/msvd/val/annotations.json
python -m scripts.check_dataloader --ann_path data/processed/msvd/test/annotations.json


```
## 实验步骤
- val: `pending=1` 且 `fail=1` 说明样本只剩1条需要补，抽帧失败
- test:`with_frames=25/26 (96.2%)`

### 精确查看三段覆盖率 & 找出失败样本




## 实验结果
- train ≈ 96%
- val/test ≥ 90%

Dataloader 验证结果 
```c:
[WARN] Dropped 7 samples without frames. kept=196
---- Batch 0 ----
video: <class 'torch.Tensor'> torch.Size([2, 8, 3, 224, 224])
caption_ids: <class 'torch.Tensor'> torch.Size([2, 32])
video_id: ['0lh_UWF9ZP4_178_182', '1pw5ZdRhiig_50_59']
---- Batch 1 ----
video: <class 'torch.Tensor'> torch.Size([2, 8, 3, 224, 224])
caption_ids: <class 'torch.Tensor'> torch.Size([2, 32])
video_id: ['8HB7ywgJuTg_131_142', '6eokrw6_bjU_1_9']
---- Batch 2 ----
video: <class 'torch.Tensor'> torch.Size([2, 8, 3, 224, 224])
caption_ids: <class 'torch.Tensor'> torch.Size([2, 32])
video_id: ['-mAoVOhKy0c_4_9', '5JSbxHECb-I_97_110']


```