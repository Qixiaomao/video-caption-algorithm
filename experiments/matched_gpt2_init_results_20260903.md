# 严格匹配的 GPT-2 初始化对照结果（2026-09-03）

## 结论

本次不支持“换回原始 GPT-2 就能解决模板化”，也不支持将剩余问题
全部归咎于 Stage3。两个完全重新初始化视觉投影/mapper 的模型仍未通过
预设的生成多样性闸门。

结果存在取舍：Stage3 组的参考字幕 NLL 更低、最大模板占比更低，
但原始 GPT-2 组的正确/错配视频 NLL 差距更大。不能把单一敏感度指标
等同于生成质量，也不能把这组结果解读为 Stage3 完全无害或必然有害。

所有结果仍为 `paper_ready=false`。本次没有修改旧 checkpoint，没有
test670 评分，没有更换模型架构，也没有追加结果驱动的训练轮数。

## 1. 预先锁定的方案

方案文件：`experiments/matched_gpt2_init_protocol_20260903.md`。

- A：原始 GPT-2，本地 HF snapshot `607a30d783dfa663caf39e06633721c8d4cfcd7e`。
- B：保留的 `checkpoints/gpt2_lm_stage3/best`。
- 仅 GPT-2 权重来源不同；架构、dropout、tokenizer 输出、ViT、视觉初始权重相同。
- GPT-2 与 ViT backbone 全冻结；仅训练投影和 mapper，共986,368个参数。
- 不使用旧 mapper/v5 权重，不加 InfoNCE；采用修复后的 caption CE 监督。
- 每组完整 train1200、5轮、750步；seed123，batch8，8帧，224x224，max_len64。
- 每视频每轮随机取一条字幕：每组共6,000次视频呈现，**不是每轮遍历全部48,774条字幕**。
- 所有训练字幕都能保留 EOS；没有截断。两组对全部训练和验证字幕的分词一致。
- AdamW，LR3e-4，weight_decay0.01，无调度器、AMP、梯度累积、早停。
- 主比较使用共同的最后一步750，不挑选各自最佳验证轮次。
- 先完成单独两步 smoke，再从相同初始权重重新开始正式训练。

训练入口 source commit：`d32e569`；评估/汇总 source commit：`9c3c6ca`。

## 2. 对照条件验收

- 两组750步的视频ID、字幕token、实际视频输入张量、Python/CPU/CUDA
  前向前随机状态逐步完全一致，两个完整 trace 文件也字节相同。
- trace SHA-256：
  `2870ce61141a0b342a697c96e4919a243c99f400029cea0cc6ca561e519b90fb`。
- 每组每轮150个batch，恰好覆盖1,200个不同视频，每视频出现一次。
- 共用视觉初始化 tensor-state SHA-256：
  `156c20ef40377a5c6eec959179b3dd6c08e48939183fadd5f0081d4b648222a0`。
- 两组 optimizer 均为空状态新建；投影和 mapper 均更新。
- 训练前后 GPT-2 与 ViT backbone 哈希不变。
- 独立从最终磁盘 checkpoint 重新加载核查：每组148个 GPT-2 参数张量
  与对应原始文件逐一一致，绑定的 LM head 也与 WTE 一致；294个冻结
  encoder state entries 全部匹配共同初始化（包含模块别名）。
- 原始 GPT-2 文件额外含12个旧版非训练 causal-mask buffer，已单独
  验证为下三角掩码；不是额外训练参数或未加载权重。
- 数据、源权重和训练源文件在运行期间未变；54项单元/回归测试通过。

| 项目 | A：原始 GPT-2 | B：Stage3 GPT-2 |
|---|---:|---:|
| 优化步数 | 750 | 750 |
| 训练及轮末验证耗时 | 2291.4秒 | 1896.7秒 |
| 峰值 allocated 显存 | 1725.6 MiB | 1727.6 MiB |
| 峰值 reserved 显存 | 1972 MiB | 1938 MiB |

耗时仅是运行记录，不是标准化速度对比；两组顺序运行，不能将温度、
缓存和机器状态造成的耗时差当成模型效率增益。RTX3050 4GB 全程无 OOM。

## 3. 训练曲线（固定首条验证字幕）

| 轮次/步数 | A train loss | A val loss | B train loss | B val loss |
|---|---:|---:|---:|---:|
| 1 / 150 | 4.8276 | 3.4025 | 3.7811 | 3.4461 |
| 2 / 300 | 3.7040 | 3.4315 | 3.1884 | 3.4337 |
| 3 / 450 | 3.5293 | 3.2651 | 3.1075 | 3.3018 |
| 4 / 600 | 3.4294 | 3.1743 | 3.0341 | 3.1946 |
| 5 / 750 | 3.3630 | 3.3058 | 2.9953 | 3.2546 |

验证 loss 并非单调下降，不能宣称已充分收敛。此处是训练器按batch平均
的固定首条字幕 loss；下文主端点使用全部参考字幕、逐参考再逐视频等权，
两者不是同一个汇总口径。

## 4. val100 主端点

两组均完成100视频、全部4,290条参考字幕、native/repeat/shuffled/zero
四条件。无参考字幕截断；同一seed、donor映射、分词和解码设置。
每组400次视频-条件评估、17,160次参考字幕-条件似然评估。

| 指标 | A：原始 GPT-2 | B：Stage3 GPT-2 |
|---|---:|---:|
| 非空描述 | 100/100 | 100/100 |
| 不同的完整句子 | 24/100 | 23/100 |
| 最大模板占比 | 32% | 24% |
| 多样性闸门 | 未通过 | 未通过 |
| Native NLL，越低越好 | 3.0642 | 2.9380 |
| Native EOS NLL，越低越好 | 0.2363 | 0.1913 |
| Native 非EOS NLL，越低越好 | 3.4681 | 3.3347 |
| 错配前缀 NLL 增量 | +0.4948 | +0.3567 |
| 增量的固定donor 95%区间 | [0.4135, 0.5796] | [0.3017, 0.4117] |
| 排除首词后的增量 | +0.5553 | +0.3964 |
| 正确前缀得分更好的视频数 | 92/100 | 94/100 |
| 错配前缀后完整句子改变比例 | 81% | 94% |
| Triage BLEU-4，非正式论文指标 | 34.2724 | 43.7541 |

预设闸门要求：非空100%、不同句子比例至少30%、最大模板占比不超过25%。
A未满足后两项；B满足最大模板约束，但未满足句子多样性约束。
闸门本身也不等同于逐视频语义正确率。

两组最常见句子都是 `a man is playing a guitar`，分别32次与24次。
A的第二常见句子为 `a dog is playing`（14次）；B为
`a man is dancing`（13次），其次是 `a man is cutting a piece of meat`（12次）。

## 5. 配对差异：B减A

均为10,000次逐视频配对 bootstrap，条件于单一训练seed和固定donor。
不包含重新训练或更换donor的不确定性，不作多重比较校正。

| 差异 | 均值 | 95%区间 |
|---|---:|---:|
| Native NLL | -0.1261 | [-0.1648, -0.0884] |
| Native EOS NLL | -0.0450 | [-0.0659, -0.0247] |
| Native 非EOS NLL | -0.1334 | [-0.1794, -0.0885] |
| 错配前缀惩罚 | -0.1381 | [-0.1885, -0.0901] |
| 排除首词后的错配惩罚 | -0.1589 | [-0.2174, -0.1035] |

非EOS NLL为端点评估前加入的描述性分解，不是追加的主验收指标。
本次未观察到 B 在 EOS 预测上更差，因此历史脚本缺少显式EOS监督的
代码风险，不能直接升级为这次恢复后模型模板化的原因。

错配惩罚更大不自动意味着生成更好：A对替换视频的惩罚较大，而B的
native参考字幕似然、最大模板占比、triage BLEU更好；这正是本次结果的取舍。

## 6. 能得出的判断与边界

1. 在本协议下，去掉 Stage3 **没有消除模板化**。Stage3不是该现象出现的必要条件。
2. 两组都完全去掉了旧 mapper 初始化，却仍未过闸门；不能再用旧权重
   遗留问题单独解释当前训练方案的全部限制。
3. Stage3权重有影响，但不是一致负效应；本次不能给出“Stage3就是根因”的结论。
4. 当前数据/监督链能学习到视频配对信息，但可靠自由生成仍未恢复。
5. 这不是充分收敛、多seed或全面解冻试验；仍不能排除训练预算、
   条件映射能力、caption目标与自由生成之间差异等因素。

建议下一步做一个小而可判定的16视频过拟合检查，验证当前修复链路
能否在极小、固定配对集合上学会逐视频生成正确描述。若能，重点转向
泛化/训练预算/目标；若不能，继续核查监督、优化和训练-生成一致性。
这个建议尚未执行，不新增模型或自动追加训练。

## 7. 产物与复现

训练/匹配结果：`runs/matched_gpt2_init_20260903/result.json`。
完整逐步轨迹与checkpoint在该目录的 `A_base/`、`B_stage3/`。

| 产物 | SHA-256 |
|---|---|
| 原始GPT-2 model.safetensors | `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707` |
| Stage3 model.safetensors | `a37dc482789e8ee4068deef308aa2793719849664054ce82db00f92efd489dcd` |
| A last.pt | `7bbb960f3867cb3f3083c7673acc4b85a583e52c101aa065a3998c76d0e89638` |
| B last.pt | `cb6ff7107571f2cb633f5c7dc91f1b6352edc90eb3397b98abf113a721de94c5` |
| A val100报告 | `fabc6d1897d9ba88b4a2aafaa6a5d2d594855979e23989539512366064078d9f` |
| B val100报告 | `38e52fc87be4739c4ed4cb942d328f569d4c8d8b7074a6bba2af5fcda887720b` |
| 配对汇总报告 | `041bd658ef81fa7af900b4b157a009546854d03eecb12f0c6e5c6d82b57a22d7` |

原始JSON和checkpoint保存在本机并由Git忽略，本摘要和执行代码版本化。

```powershell
.\.venv\Scripts\python.exe scripts/run_matched_gpt2_init.py --output-root runs/matched_gpt2_init_20260903
.\.venv\Scripts\python.exe scripts/diagnose_prefix_sensitivity.py --checkpoint runs/matched_gpt2_init_20260903/A_base/last.pt --expected-checkpoint-sha256 7bbb960f3867cb3f3083c7673acc4b85a583e52c101aa065a3998c76d0e89638 --limit 100 --preset core --output reports/matched_gpt2_init_A_base_val100_20260903.json
.\.venv\Scripts\python.exe scripts/diagnose_prefix_sensitivity.py --checkpoint runs/matched_gpt2_init_20260903/B_stage3/last.pt --expected-checkpoint-sha256 cb6ff7107571f2cb633f5c7dc91f1b6352edc90eb3397b98abf113a721de94c5 --limit 100 --preset core --output reports/matched_gpt2_init_B_stage3_val100_20260903.json
.\.venv\Scripts\python.exe scripts/summarize_matched_gpt2_init.py --trial-root runs/matched_gpt2_init_20260903 --a-report reports/matched_gpt2_init_A_base_val100_20260903.json --b-report reports/matched_gpt2_init_B_stage3_val100_20260903.json --output reports/matched_gpt2_init_summary_20260903.json
```

命令会拒绝覆盖既有试验/报告；复跑必须使用新的输出路径。
