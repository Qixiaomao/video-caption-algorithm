# 今日实验总结 (2025-10-01)

## 数据准备
- 使用多进程抽帧脚本补齐了train split 的帧数据：

```c:
python -m scripts.extract_frames_mp --splits train --fps 2 --workers 4 --only-missing
```

- 结果：
```c:
[DONE] split=train ok=169 fail=7
[COVERAGE] train: with_frames=196/203 (96.6%)
```
- 覆盖率从 13.3% -> 96.6% 可用样本大幅提升

## Dataloader 修复
- 修改了 `src/data/data_loader.py`:
   - _sample_indices: 永远返回固定`num_frames` (不足时循环补齐，多余时均匀采样)
   - getitem:增加补齐/截断逻辑，确保每个样本输出的`video` 形状一致
   - 验证指令：
     ```c:
    python -m src.cli.train --ann_path data/processed/msvd/train/annotations.json --batch_size 2 --num_frame 8 --image_size 224 --max_len 32 --seed 123

    ```
    - 输出确认每个batch[2,8,3,224,224] , 再无shape mismatch 报错。

## 干跑训练(Dry Run)
 - 使用极简 `SimpleAlignModel`(视频均值+文本均值+CosineEmbeddingLoss) 测试训练链路
 - 指令：
   ```c:
   python -m src.cli.train --ann_path data/processed/msvd/train/annotations.json ^
  --batch_size 2 --num_frame 8 --image_size 224 --max_len 32 --seed 123 ^
  --epochs 1 --max_steps 50 --lr 5e-4
   ```

 - 控制台日志：
 ```c:
step 0001 | loss 1.0457
step 0010 | loss 0.9351
step 0020 | loss 0.6001
step 0030 | loss 0.5465
step 0040 | loss 0.4904
step 0050 | loss 0.7053
 ```

 - Loss曲线：整体下降，50步后出现小幅抖动(batch小+lr偏高的正常波动)
 


## 今日结论
1. 数据准备就绪：train split 覆盖率已达 96.6%，val/test 待补齐
2. Dataloader 稳定：支持固定帧数采样，batch 对齐成功。
3. 训练链路打通：模型能跑满50步，loss明显下降，说明数据管道和优化器流程正确
4. 可视化建立：实验结果已存档(csv+png),方便后续比较。

## 下一步计划

补齐 val/test 覆盖率，验证集也能正常用。

跑一轮 长 dry run（200~300 steps），观察 loss 稳定性。

替换 SimpleAlignModel → 正式模型骨架（ViT/TimeSformer + GPT2）。

增加评估指标（BLEU/ROUGE 或简单的 cosine 对齐 score），让实验更直观