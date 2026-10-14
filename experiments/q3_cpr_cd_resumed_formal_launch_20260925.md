# Q3 C/D 剩余正式训练启动记录（2026-09-25）

## 启动状态

2026-09-25 12:01:26（Asia/Shanghai）启动隐藏后台进程，入口为
`scripts/finish_cpr_cd_matrix.py`，输出目录为
`runs/q3_cpr_cd_resumed_formal_20260925`。启动器 PID 28200；当时实际 Python
训练进程 PID 39732。进程号只供本次运行诊断，重启后不应再据此识别任务。

启动前：注册输入与断点恢复预检通过，CPR 相关测试 16/16 通过；GPU 空闲，D 盘
剩余约 204 GiB；test670 未读取。原注册 runner SHA-256 为
`af52b5e2f1adf32f94244bd59f85a210703b93acdf6d16205b07a9da2c2b1f5a`。

12:02:20--12:02:28 的两次采样表明 seed124-C 从 step6045 增至 step6060，
8 秒新增 15 步，GPU 利用率 100%，显存约 2587 MiB。恢复入口持续核对暂停前
的原始轨迹与损失；到原日志结束后继续与注册历史轨迹核对。

12:07:21 已到 epoch44、step6537，越过暂停前最后一条完整指标 step6512。
核对未报错，进程与 GPU 计算仍正常；后续训练继续由后台脚本自动推进。

## 自动执行顺序

1. 从 epoch40 checkpoint 续跑 seed124-C 至 epoch66，合并并核对完整9900步轨迹；
2. 运行 seed124-D；
3. 运行 seed125-C、seed125-D；
4. 核对 C/D 配对轨迹，读取已完成的 seed123-C/D 结果；
5. 在 val100 上执行预注册 CPR 保留门槛，写入 `completion.json`。

如果任一校验失败，脚本会退出并保留已经写出的产物，不会生成
`completion.json`。不能把不存在的完成记录解释为实验通过。

## 状态与边界

- 标准输出：`runs/q3_cpr_cd_resumed_formal_20260925.stdout.log`
- 错误输出：`runs/q3_cpr_cd_resumed_formal_20260925.stderr.log`
- 成功标志：`runs/q3_cpr_cd_resumed_formal_20260925/completion.json`
- 运行机器交流供电时的系统睡眠设置为“从不”；直流供电设置为180秒，长训需
  保持电源连接。
- 此次只使用 train1200/val100，test670 继续封存；最终 CPR 结论只能在全部
  三随机种子完成并形成 `completion.json` 后报告。
