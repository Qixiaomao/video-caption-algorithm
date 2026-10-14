# Q3 C/D seed124-C 恢复入口验证（2026-09-25）

## 结果

已实现独立恢复入口 `scripts/resume_cpr_cd_matrix.py`，原注册 runner 的 SHA-256
仍为 `af52b5e2f1adf32f94244bd59f85a210703b93acdf6d16205b07a9da2c2b1f5a`。
恢复入口从 2026-09-12 的 seed124-C epoch40 checkpoint 读取模型、优化器、RNG、
训练契约和注册 schedule，并在新目录重放 step6001--6016。

验证结果：16/16 条训练轨迹与停机前日志逐字段一致，包括视频与字幕、视觉特征
hash、Python/CPU/CUDA RNG hash；16/16 条训练指标逐值相等，包括 caption CE、
CPR 与总损失。测试集 test670 未读取。验证后 GPU 显存回到 0 MiB。

| 校验物 | SHA-256 |
|---|---|
| 恢复 checkpoint | `8217b670c4f791023e6b7c3680f4ba11b27f0b0efa27f7c887912eada9681ce4` |
| 恢复入口（验证后） | `f2a1c8859b77ea12092f15c8adebc1be0a3300e4ec588b42fc54ac152110849d` |
| 验证报告 | `629abcce35fe4c0f6b173ed466052c34d985e5ff0bcf2a4083de23c96c79f23b` |

验证报告：`runs/q3_cpr_cd_resume_verify_20260925/verification.json`。

## 恢复入口的保护条件

- 只允许写入尚不存在的新输出目录，不改动 2026-09-12 暂停目录。
- 核对暂停 checkpoint、trace、指标日志、seed123 完整结果和全部注册输入/hash。
- 核对 epoch40/step6000、模型参数、优化器、训练契约、数据契约与 schedule。
- 完整运行模式会把前6000步原始记录和续跑记录合成一条轨迹，要求最终
  9900步 trace 的 SHA-256 与历史 A 臂完整轨迹一致。
- 完整运行的显存峰值与耗时标注为“续跑段”，不会伪装成整个66轮的统计。

CPR 相关单元测试与注册矩阵测试共 16/16 通过。`pytest` 不在项目虚拟环境中，
测试使用原有 `unittest` 执行。

## 当前边界与下一步

这次只验证了从 epoch40 恢复后的前16步；**seed124-C 的剩余长训练尚未启动**，
seed124-D、seed125-C/D 也仍未开始。恢复正确性已得到直接数值证据，但 epoch66
结果仍需完成整段运行后才能成立。

下一次运行 seed124-C：

```powershell
.\.venv\Scripts\python.exe -u scripts\resume_cpr_cd_matrix.py --complete --output-root runs\q3_cpr_cd_resumed_formal_20260925
```

输出目录必须是一个新目录。seed124-C 完成并核对完整 trace 与最终 checkpoint
后，再运行 seed124-D 和 seed125-C/D，并执行预注册 CPR 保留门槛。在门槛冻结前
继续封存 test670。
