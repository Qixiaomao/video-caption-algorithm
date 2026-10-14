# Q3 C/D 正式长训练暂停记录（2026-09-12）

## 暂停结论

正式 C/D 矩阵于 2026-09-12 17:52 启动，于 23:03:50 主动关停。
关停前没有出现 Traceback、CUDA OOM、NaN 或 Inf，GPU 显存已由约
2671 MiB 释放至 0 MiB。

- 已完整完成：seed123-C、seed123-D；
- 正在执行时关停：seed124-C；
- seed124-C 原始日志推进到 epoch 44、step 6511；
- 最近的严格可恢复点：epoch 40、step 6000；
- 正式可恢复进度：25,800 / 59,400 steps（43.43%）；
- test670 始终未读取，当前结果均为 val100 证据。

step 6001--6511 没有对应的完整训练状态 checkpoint，明日续跑时不计入正式
结果。现有原始日志必须先留档，再从 epoch40 checkpoint 恢复模型、优化器、
RNG 和注册训练契约。

## 已完成的 seed123 结果

| seed | arm | epoch/step | NLL | mean max token F1 | grounding gap | caption gate | T2V R@1 | T2V R@5 | T2V R@10 |
|---:|---|---:|---:|---:|---:|---|---:|---:|---:|
| 123 | C | 66 / 9900 | 2.892322 | 0.693692 | 1.404954 | pass | 10.1632% | 35.4312% | 50.0932% |
| 123 | D | 66 / 9900 | 2.893516 | 0.699200 | 1.371954 | pass | 18.0420% | 52.5641% | 68.0653% |

两臂均完成四个注册 checkpoint（epoch 5/20/40/66），并通过：

- historical trace prefix match；
- C/D paired trace match；
- frozen backbone/decoder hash 不变；
- final checkpoint 严格重载；
- source/data unchanged 检查。

这只是 seed123 的单种子验证结果，不能提前替代三随机种子的预注册判断。

## seed124-C 的恢复点

epoch40 的 val100 中间结果：

| NLL | mean max token F1 | grounding gap | caption gate | T2V R@1 | T2V R@5 | T2V R@10 |
|---:|---:|---:|---|---:|---:|---:|
| 2.878760 | 0.711594 | 1.327995 | pass | 6.0839% | 28.3450% | 45.0816% |

恢复 checkpoint：

`runs/q3_cpr_cd_formal_20260912/seed124/C/checkpoint_epoch040.pt`

- SHA-256：`8217b670c4f791023e6b7c3680f4ba11b27f0b0efa27f7c887912eada9681ce4`
- checkpoint version：3；
- progress：epoch 40、step 6000、epoch_complete=true；
- 已包含 model、optimizer、RNG、loader state、training contract 和 dataset contract。

暂停时保留的未封存日志：

- `train_metrics.jsonl` SHA-256：`112cf971c93dc3baeaf99541dc1621cee28798c7e9764591159fb6a732dbe14c`
- `trace.jsonl` SHA-256：`caa5033b2f8e4f69946f5a9fb05ebff09aed22593e8ca8e298fea951290f7b6a`

## 完整结果文件校验

- seed123-C `result.json`：
  `340612740c2b29134eafd00917e46aa9e8ac3d2591e1a2eb13e3024daadec366`
- seed123-D `result.json`：
  `df7538d8528085f993bbb20863938f9f36822ec27f6e7b6940b205caa80af6d7`
- seed123 `pair_result.json`：
  `7c45a0e5316d08a3ed7940c4c0387dbbd706ed1cd71f5ad5460d26bab9f6ae23`

## 明日续跑顺序

1. 不修改原注册 runner 和既有完整结果；新增并测试独立 resume adapter。
2. 将 seed124-C 的 step 6001--6511 原始日志只读归档，不直接覆盖或丢弃。
3. 严格加载 epoch40 checkpoint，验证 checkpoint/训练契约/数据/hash 后，从
   epoch41 和注册 schedule 的对应偏移继续。
4. 完成 seed124-C 的 epoch66，合并前后 trace 后再次执行 historical/C-D trace
   一致性检查。
5. 依次执行 seed124-D、seed125-C、seed125-D。
6. 三个种子全部完成后才执行预注册 CPR 保留门槛；通过前不读取 test670。

不得直接对现有输出目录重新运行原始命令：原 runner 会拒绝覆盖，而且原 runner
没有注册的断点续训入口。
