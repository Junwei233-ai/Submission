

## 0. 最终方法与结果(先读这一节)

**最终方法:GRU 方案 B**(停车点 MLP + GRU,带停止头和锁存,`--features park`);**baseline:GRU v3**
(`--features bearing`,含手写的世界方位角特征);**参照:专家控制器**。
所有命令都在**仓库根目录**运行。第 5 节及以后的数字是开发阶段的历史记录,以本节为准。

### 0.1 为什么要重新评估

开发阶段的测试集被使用了 3 次,而且方案 B 是在分析测试集失败之后设计的(第 5.1、8 节),
旧测试集上的数字因此不再是无偏估计。本节的协议在看到任何结果前固定:

1. 用同一份数据(1500 个场景,种子 42)重新训练 v3 和 B,划分为训练 973 / 验证 208 / 测试 210
   (本地专家成功 1391 条,比开发阶段少 1 条)。
2. 两个模型**各自**在全部 208 个验证场景上网格搜索 Kp/Kd/lead(12 组),规则:1 cm 成功最多 →
   平均用时最短 → 力矩限幅比例最低。
3. 在旧测试集上核对能否复现开发阶段报告的数字(误差 ±3 个场景以内算复现)。
4. 模型冻结后,**新采集 300 个从未用过的场景(种子 2026)**作为最终测试集,只运行一次;
   同一批场景上同时运行 B、v3 和专家,做配对比较。
5. 鲁棒性与分布外泛化同样在冻结后、在新场景上运行。

### 0.2 一键复现

```bash
PY=/path/to/RoboEnv/python      # 换成你自己的 Python(需要 torch、pybullet、pinocchio、simulation_and_control)
$PY collect_data.py --n 1500 --seed 42 --workers 6 --output artifacts/final_run/collection   # 训练数据
$PY collect_data.py --n 300 --seed 2026 --workers 6 --output artifacts/final_run/fresh_test_seed2026  # 新测试场景
SOURCE=artifacts/final_run/collection PY=$PY bash run_final_pipeline.sh artifacts/final_run 9   # 训练、选增益、复现核对
PY=$PY bash run_evaluation_suite.sh artifacts/final_run 18 60                                     # 新测试、鲁棒性、分布外
PY=$PY bash run_bc_analyses.sh artifacts/final_run 18 3                                           # 种子、学习曲线、重力消融、保持
```

输出:`reproduction.md`(复现核对)、`grid_*/grid.json`(增益网格)、`suite/summary.md`(最终结果表)、
`analysis/summary.md`(第 0.9 节的补充分析)。
所有脚本拒绝覆盖已有的输出目录;中断后重跑会跳过已完成的部分。

### 0.3 训练与增益

| | v3(baseline) | **B(最终)** |
|---|---|---|
| 最好的一轮 / 实际轮数(上限 80,patience 8) | 68 / 76 | 78 / 80 |
| 锁存长度(验证集开环选出) | 2 | 1 |
| 选出的增益 Kp/Kd/lead | 100/10/0.1 | 200/30/0.2 |
| 验证集 1 cm 成功 | 205/208 | 207/208 |
| 12 组增益的验证成功范围 | 189–205(10 组 ≥ 198) | 193–207(10 组 ≥ 206) |
| GRU 参数 | 60,186 | 60,186 + 停车点 MLP 17,540 |

结果对增益不敏感:成绩主要来自学到的策略,而不是调出来的 PD。B 在第 78 轮达到最好,
接近 80 轮上限,可能尚未完全收敛。

注:这两个冻结模型的 `gru.json` / `gru.npz` 元数据中 `labels` 字段是旧的说明文字(写着 "no stop/phase"),
因为训练在修正 `staged_gru_train.py` 的这段文字之前就已完成;模型实际带停止头,以同一份元数据中的
`stop_head: true`、`feature_set`、`arm_output: matrix` 为准。

### 0.4 复现核对(旧测试集,210 个专家成功场景)

| 项目 | 增益 | 本地 | 开发阶段报告 | 复现 |
|---|---|---|---|---|
| v3 | 自己选的 100/10/0.1 | 204 | 205(当时的增益 50/30/0.2) | ✅ |
| v3 | 100/30/0.1 | 203 | 204 | ✅ |
| B | 自己选的 200/30/0.2 | 208 | 206(沿用 v3 的增益 50/30/0.2) | ✅ |
| 专家 | — | 209 | — | 参照 |

验证集上 B 在 50/30/0.2 下为 206/208,开发阶段报告 202(本地更好)。

### 0.5 最终结果:新测试集(种子 2026,300 个场景,未筛选)

| | 1 cm 成功 | 95% Wilson 区间 | 专家成功的 286 个 | 专家失败的 14 个 |
|---|---|---|---|---|
| **GRU 方案 B** | **296(98.7%)** | 96.6–99.5% | 282 | 14 |
| GRU v3 | 289(96.3%) | 93.6–97.9% | 277 | 12 |
| 专家 | 293(97.7%) | 95.3–98.9% | 286 | 7 |

配对精确 McNemar:B vs 专家 7/4(只有 B 成功 / 只有专家成功),p = 0.55;B vs v3 10/3,p = 0.092。
两者都不显著:B 与专家**持平**,比 v3 略好。B 的 4 个失败中 3 个从未锁存(底盘一直没判定停好),
1 个已锁存但末端停在 6 cm 处。

### 0.6 鲁棒性与泛化(冻结后运行,B 与专家使用相同场景和相同扰动)

| 条件 | n | B | 专家 | 配对 p |
|---|---|---|---|---|
| 观测噪声 σ = 0.003(m / rad;关节速度 10σ) | 60 | 59 | 60 | 1 |
| 观测噪声 σ = 0.01 | 60 | 59 | 60 | 1 |
| 机械臂力矩只输出 80% | 60 | 59 | 60 | 1 |
| 底盘停好 1 s 后推末端 30 N × 0.2 s | 60 | 59 | 60 | 1 |
| t = 2 s 推底盘 20 N × 0.2 s | 60 | **56** | 38 | 4e-5 |
| 分布外:目标更远(max(|x|,|y|) ∈ (1.6, 2.2]) | 40 | 40 | 40 | 1 |
| 分布外:目标更高(z ∈ (1.30, 1.40]) | 40 | **40** | 15 | 6e-8 |
| 分布外:目标更低(z ∈ [0.03, 0.10)) | 40 | 29 | 30 | 1 |
| 分布外:起点更宽(底盘 ±1.2 m,关节 ±0.30 rad) | 40 | 38 | 39 | 1 |

- 噪声、力矩减弱、推末端:B 都只失败同一个场景 `scene_2026_0013`(无扰动时也失败,从未锁存),
  其余 59 个全部成功,性能不下降;PD 层和 GRU 的闭环反馈吸收了扰动。
- 推底盘:专家的导航状态机被推偏后无法恢复,拖到 25 s 超时,底盘偏离停车点中位 1.6 m;
  B 的 GRU 每 10 ms 根据当前位姿重新决策,能够纠正。仿真中轮子侧向摩擦很低
  (`anisotropicFriction=[1, 0.01, 1]`),20–40 N 已能推出米级滑移。
- 高目标:专家的手写规则把目标投影到离肩 0.85 m 的球面上,最近仍差 1.8–5.8 cm;B 没有这条规则,
  学到的速度场可以外推。这说明 B 并非逐条复制专家规则。
- 低目标:两者都只有约 73%,与训练分布中的主要失败模式(接近关节限位、奇异位形)一致。

### 0.7 效率

| | 每个 1 kHz 控制步的计算时间 | 模型文件 |
|---|---|---|
| B | 87 µs | 616 KiB |
| v3 | 82 µs | 477 KiB |
| 专家(IK + PD + 重力) | 82 µs | — |

时间包含特征计算、每 10 步一次的 GRU 更新(NumPy,无需 PyTorch)、PD 与重力项;远低于 1 ms 控制周期。

### 0.8 专家隔离

学习到的策略运行时**不加载**专家模块:`staged_rollout.py` 只在 `--base expert` 或 `--arm expert`
时导入 `robotnik_arm_control.py`;两个用到的工具函数(四元数转 yaw、固定手指)在 `sim_utils.py`
中独立实现(`test_sim_utils.py` 验证与模板一致)。每次学习策略的运行都在 `result.json` 中记录
`expert_module_loaded: false`,若加载会直接报错。

### 0.9 补充分析(冻结后运行,不改变冻结模型和增益)

由 `run_bc_analyses.sh` 生成,汇总在 `analysis/summary.md`。验证集分析都用 B 选出的增益 200/30/0.2。

**训练种子**(只换 GRU 的随机种子;划分、停车点 MLP、增益不变;验证集 208 个场景)

| 种子 | 1 cm 成功 | 最好的一轮 | 锁存长度 |
|---|---|---|---|
| 42(冻结模型) | 207 | 78 | 1 |
| 1 | 207 | 52 | 2 |
| 2 | 208 | 78 | 3 |

结果不依赖某个幸运的种子。

**学习曲线**(停车点 / 停止 MLP 和 GRU 全部在训练轨迹的嵌套子集上重新训练;验证集 208 个场景)

| 训练数据 | 轨迹数 | 1 cm 成功 | 验证损失 轮速 / 机械臂 / 停止 | 失败中:未锁存 / 锁存后没到 | 停车偏差中位 | 训练时间 |
|---|---|---|---|---|---|---|
| 10% | 97 | 157 | 0.283 / 0.858 / 0.162 | 24 / 27 | 15.2 cm | 6 min |
| 25% | 243 | 193 | 0.156 / 0.465 / 0.108 | 5 / 10 | 10.8 cm | 11 min |
| 50% | 486 | 195 | 0.086 / 0.249 / 0.042 | 1 / 12 | 8.7 cm | 19 min |
| 100% | 973 | 207 | 0.054 / 0.163 / 0.021 | 1 / 0 | 7.5 cm | 31 min |

三个离线验证损失随数据量单调下降,闭环成功率随之上升。数据少时的失败主要来自**底盘**:
10% 时一半失败是从未锁存(停止判断没学好);25–50% 时锁存后的失败末端误差中位 0.55–0.64 m,
即底盘停在了机械臂够不着的地方(停车点学得不准),而不是机械臂速度场的问题。

**PD 层的重力项**(`--gravity zero`,验证集,配对)

| 增益 Kp/Kd/lead | 有重力项 | 去掉重力项 | 配对不一致 (p) |
|---|---|---|---|
| 200/30/0.2(选出的) | 207 | 207 | 0/0 (1) |
| 50/30/0.1 | 203 | 57 | 147/1 (8e-43) |

示范数据中肩关节(第 2 关节)力矩中位 17.7 N·m,主要是重力。参考关节角对 `M·e` 积分,
起到积分作用;只要 Kp × lead 大于约 18 N·m,PD 自己就能扛住重力,模型重力项不是必需的。
50 × 0.1 = 5 N·m 时扛不住,机械臂下垂(最终误差中位 32 cm)。

**成功后再保持 2 s**(`--hold-time 2`,新测试集 300 个场景)

模板的成功判定是 8 s 后**第一次**进入 1 cm 就结束。这里成功后继续仿真 2 s,检查是否保持住。

| 控制器(增益) | 1 cm 成功 | 2 s 内一直在 1 cm 内 | 最后 1 s 在 1 cm 内 | 最后 1 s 最大误差 中位 / p95 |
|---|---|---|---|---|
| **B**(200/30/0.2) | 296 | **286** | 296 | 2.2 / 3.6 mm |
| 专家 | 293 | 286 | 293 | 2.4 / 3.4 mm |
| v3(100/10/0.1) | 289 | 171 | 228 | 5.7 / 19.0 mm |
| 诊断:v3 改用 100/30/0.1 | 289 | 278 | 289 | 2.3 / 4.1 mm |
| 诊断:B 改用 100/10/0.1 | 296 | 144 | 217 | 6.1 / 23.9 mm |

- B 与专家保持能力相同(配对 8/8,p = 1);B vs v3 为 122/7(p = 3e-28)。
- v3 的问题来自**增益**而不是策略:两个策略只要用 Kd = 10,都会在目标附近 0.5–1 Hz 来回摆
  (2–28 mm);改成 Kd = 30 就和专家一样稳。换增益**不改变** 1 cm 成功数。
- 原因:增益网格的规则在成功数相同时选"平均用时最短",而"第一次进入 1 cm"看不到阻尼不足,
  于是偏向冲得快、阻尼小的增益。B 选到 Kd = 30 是因为那一组恰好多成功 1 个场景。
- 这是在测试集上发现的,所以只作为诊断,不据此修改 v3 的增益或主结果。改进办法:把"保持"
  写进选择规则或成功判定。

# COMP0245:移动机械臂的行为克隆

任务:根据测量值 `base_pos, base_bearing, q_mes, qd_mes` 和目标位置 `TARGET_EE_POS`,
输出 `wheel_cmds`(4 个轮速)和 `torque_arm`(7 个关节力矩),让末端到达目标。

所有模型**只用录制的专家轨迹训练**:不读取专家控制器的内部变量(`phase`、`q_des`、`qd_des` 等),
运行时也不调用专家。GRU 的详细推导见 [GRU.md](GRU.md)。

## 1. 目录说明

### 1.1 代码(全部由我们编写)

| 文件 | 作用 | 主要输入 → 输出 |
|---|---|---|
| `collect_data.py` | 生成随机场景,并行调用专家采集脚本,汇总结果 | `--n 1500 --seed 42` → `artifacts/collection_*/`(每个场景一个 `.npz`,另有 `summary.csv`) |
| `staged_data.py` | 公共工具:从录制数据中提取**公开停车时刻**、停车位姿和底盘特征;按整条轨迹做 70/15/15 划分 | 被其他脚本 import |
| `staged_train.py` | 训练底盘部分:停车点 MLP(方案 B 使用)、轮速 MLP、停止 MLP(后两者属于四网络方案);**同时生成所有模型共用的 `split.json`**。`--train-fraction f` 只保留训练集的一部分轨迹(嵌套子集,验证/测试不变),用于学习曲线 | 数据集 → `<输出目录>/`(`policy.npz`、`split.json`、指标);最终运行为 `artifacts/final_run/base/` |
| `staged_arm_train.py` | 训练四网络方案的机械臂 MLP(输出 7×3 矩阵 M;开发阶段对照,最终方法不使用) | `split.json` → `arm.npz` |
| `staged_gru_train.py` | 训练 GRU。选项:`--stop-head`(停止输出 + 锁存)、`--features bearing\|park`(v3 或方案 B 的输入)、`--arm-output matrix\|torque`(力矩消融) | `split.json` → `gru.npz`(含训练记录 `gru.json`);最终运行为 `artifacts/final_run/gru_*/` |
| `staged_policy.py` | 所有模型的 **NumPy 运行时**(不依赖 PyTorch):`StagedBase`、`StagedArm`、`GRUPolicy`,以及输入特征函数 | 被 rollout 脚本 import |
| `staged_rollout.py` | **单个场景**的闭环仿真。底盘和机械臂可分别选择"我们 / 专家 / GRU";仿真设置与采集时相同。评估选项:扰动(`--obs-noise` 等)、`--hold-time`(成功后继续仿真并记录保持期最大误差)、`--gravity zero`(PD 层去掉重力项的消融) | 场景 JSON → `rollout.npz` + `result.json` |
| `staged_evaluate.py` | 在一个数据划分上**并行**运行 `staged_rollout.py`,汇总成功率、Wilson 区间、失败原因。`--split` 可选 train / validation / test / excluded(专家未成功的 108 个场景) | → `artifacts/staged_eval/<名字>/`(`metrics.json`、`results.json`、每个场景的日志和录像) |
| `staged_arm_grid.py` | 在验证集上网格搜索 PD 增益 Kp、Kd、lead。`--arm-model` 搜索四网络的机械臂(配专家底盘),`--gru-model` 搜索 GRU;选择规则事先固定 | → `grid.json` |
| `sim_utils.py` | 仿真工具(四元数转 yaw、固定手指),独立实现,使学习策略的运行不依赖专家模块 | 被 `staged_rollout.py` import |
| `make_ood_scenes.py` | 生成 4 类分布外场景(目标更远 / 更高 / 更低、起点更宽) | → 每类一个场景 JSON 目录 |
| `run_final_pipeline.sh` | 最终协议:训练 v3 与 B → 各自验证集网格 → 旧测试集复现核对 | → `reproduction.md` |
| `report_reproduction.py` | 把本地结果与开发阶段报告的数字对照 | → `reproduction.md` |
| `run_evaluation_suite.sh` | 冻结后评估:新测试集、鲁棒性(`staged_evaluate.py --obs-noise/--torque-scale/--push-ee/--push-base`)、分布外 | → `suite/` |
| `summarize_suite.py` | 汇总成功率、Wilson 区间、配对 McNemar、计算时间、模型大小 | → `suite/summary.md` |
| `run_bc_analyses.sh` | 冻结后的补充分析:训练种子、学习曲线、重力项消融、成功后保持 2 s | → `analysis/` |
| `summarize_analyses.py` | 汇总上述补充分析 | → `analysis/summary.md` |
| `test_sim_utils.py` | 检查 `sim_utils.quaternion_yaw` 与模板 `quaternion2bearing` 一致 | `python -m unittest test_sim_utils` |

### 1.2 作业模板与采集脚本:`robotnik_arm_task/`

| 文件 | 来源 | 作用 |
|---|---|---|
| `robotnik_arm_task.py` | 作业模板,**未修改** | 专家演示脚本:底盘导航到停车点,然后机械臂伸向目标 |
| `robotnik_arm_control.py` | 作业模板,**未修改** | 专家控制器(`base_controller`、`arm_controller`)、停止判定 `check_stop` |
| `robotnik_arm_collect.py` | 由我们从 `robotnik_arm_task.py` **复制后修改** | 无界面运行一个场景,保存轨迹 `.npz` 和成功标签(修改内容见第 2 节) |
| `configs/`、`models/` | 作业模板(RoboEnv) | 机器人配置文件、URDF 和网格 |
| `README.md` | 作业模板 + 我们补充的一段说明 | 模板的运行说明 |

### 1.3 文档

| 文件 | 内容 |
|---|---|
| `README.md` | 本文件:目录、外部代码、方法概要、结果、运行命令 |
| `GRU.md` | GRU 从采集、训练到测试的完整说明和数学推导 |

### 1.4 本地数据和结果:`artifacts/`(已在 .gitignore 中,不提交)

**最终结果:`artifacts/final_run/`**(由第 0.2 节的脚本生成)

| 目录 / 文件 | 内容 |
|---|---|
| `base/` | `split.json`(训练 973 / 验证 208 / 测试 210,所有模型共用);停车点、轮速、停止网络(方案 B 使用其中的停车点网络) |
| `gru_v3_bearing/`、`gru_B_park/` | **冻结的 v3 和方案 B**(`gru.npz` 模型,`gru.json` 训练记录) |
| `grid_v3_bearing/`、`grid_B_park/` | 各自的验证集增益网格,`grid.json` 中 `chosen` 为选出的增益 |
| `oldtest/`、`reproduction.md` | 旧测试集上的复现核对 |
| `fresh_test_seed2026/`、`fresh_split/` | 新测试场景(种子 2026)及其划分 |
| `ood_scenes/` | 4 类分布外场景 |
| `suite/` | 新测试集、鲁棒性、分布外的评估结果,汇总在 `suite/summary.md` |
| `analysis/` | 补充分析(种子、学习曲线、重力项、保持),汇总在 `analysis/summary.md` |
| `*.log` | 训练、网格和评估日志 |

本次运行的训练数据来自 `artifacts/verify_teammate_torque_20261105/collection_public/`
(`collect_data.py --n 1500 --seed 42` 采集,专家成功 1391 条)。

**开发阶段的目录**(第 5 节的历史数字来自这里,最终结果不使用):

| 目录 | 内容 |
|---|---|
| `collection_1500_seed42_20261031_013545_761934/` | 数据集:1500 个场景的专家轨迹 + `summary.csv` |
| `staged_base_v1/` | `split.json`(所有模型共用的划分);四网络的停车点、轮速、停止网络(方案 B 也使用其中的停车点网络) |
| `staged_arm_v1/` | 四网络的机械臂网络 |
| `staged_gru_v1/`、`staged_gru_v2/` | 早期 GRU(没有停止输出),只用于消融对比 |
| `staged_gru_v3_stop/` | 开发阶段的 GRU v3:停止输出 + 锁存,输入含手写的世界方位角 |
| `staged_gru_v3_park/` | 开发阶段的方案 B:用学到的停车点替代世界方位角 |
| `staged_gru_v3_torque/` | 消融:GRU 直接输出力矩 |
| `staged_eval/` | 所有评估结果,见下表 |

`staged_eval/` 中各目录:

| 目录 | 内容 |
|---|---|
| `expert_expert_val` | 专家在验证集上的表现 |
| `base_v1_ours_expertarm_val` | 我们的底盘 + 专家机械臂 |
| `arm_v1_expertbase_val*`、`arm_v1_grid_val40` | 专家底盘 + 四网络机械臂;四网络机械臂的增益网格(40 个场景) |
| `full_v1_ours_ours_val*` | 四网络完整方案(其中一个加了 25 s 超时) |
| `gru_v1_val`、`gru_v2_val`、`gru_v3_stop_val` | 各版 GRU 的验证结果 |
| `gru_v3_grid_val208` | 开发阶段 GRU v3 的增益网格(208 个验证场景,当时选出 50/30/0.2) |
| `gru_v3_park_val_kp50_kd30_lead0.2` | 方案 B 的验证结果 |
| `gru_v3_torque_val` | 力矩消融的验证结果 |
| `TEST_fournet_v1`、`TEST_gru_v3_stop` | 测试集,第一次运行(增益 100/30/0.1) |
| `TEST_gru_v3_stop_kp50_kd30_lead0.2` | 测试集,v3 使用自己选出的增益 |
| `TEST_gru_v3_park_kp50_kd30_lead0.2` | 测试集,方案 B |
| `expert_excluded`、`gru_v3_excluded*`、`gru_v3_park_excluded_*` | 专家、GRU v3、方案 B 在 108 个被排除场景上的表现 |

**其余目录**(`tcn_*`、`training_*`、`evaluation_*`、`arm_*`、`collection_wide_30*`、`diagnostic_*`、
`control_diagnosis`、`mlp_basefeat_eval`、`sanity_*`、`expert_replay_0013`、`scenarios`)都是更早的旧方案(端到端 MLP / TCN,代码不在本仓库)产生的,
与当前方法无关。

### 1.5 其他

| 路径 | 内容 |
|---|---|
| `results/final_run/` | **最终运行的结果快照**(纳入版本管理,约 3 MB):各汇总表 `suite/summary.md`、`analysis/summary.md`、`reproduction.md`;冻结模型 `gru_v3_bearing/`、`gru_B_park/`;增益网格 `grid_*/grid.json`;划分 `base/split.json`、`fresh_split/split.json`;各次评估的 `metrics.json` 和 `config.json`;`public_data_provenance.json`(训练数据只去掉了 `phase`、`time` 字段,其余数组逐位不变)。逐场景的录像和日志不包含 |
| `.gitignore` | 忽略数据、缓存、日志 |
| `__pycache__/` | Python 自动生成的缓存,可以删除 |

## 2. 外部代码及修改情况

| 外部代码 | 用在哪里 | 是否修改 |
|---|---|---|
| 作业模板 `robotnik_arm_task.py`、`robotnik_arm_control.py`、`configs/`、`models/` | 专家控制器、仿真配置 | 未修改 |
| 作业模板 `robotnik_arm_task.py` → 我们的 `robotnik_arm_collect.py` | 采集专家轨迹 | **修改**:场景参数(目标、底盘起点和朝向、机械臂初始角)由 `--scene` 传入;无界面运行;保存轨迹(`inputs`、`actions`、`ee_pos`、`phase`、`time`);把 `check_stop` 的结果记为 `success` / `stalled` / `step_limit`;另外要求最后一步误差也 < 1 cm 才记为成功(比模板更严格,见第 6 节) |
| RoboEnv 的 `simulation_and_control` 库 | PyBullet 仿真接口 `SimInterface`、`PinWrapper`、`MotorCommands` | 未修改 |
| pinocchio | 正运动学(末端位置)、重力项 g(q) | 直接调用 |
| PyTorch | 训练 | 直接调用 |

`staged_rollout.py` 中的仿真设置(地面摩擦、轮子接触参数、电机增益、手指角度、站位规则)复制自
`robotnik_arm_collect.py`,以保证评估和采集条件一致。站位规则只在运行**专家**时使用。

除上表外,仓库根目录下的所有 `.py` 都是我们从零编写的。

## 3. 数据

- **数据集**:1500 个随机场景(底盘起点 xy ±0.6 m、朝向任意、机械臂初始角为参考姿态 ±0.15 rad、
  目标 xy ±1.6 m、高度 0.1–1.3 m),其中按采集脚本的标准,专家成功 1391 条(最终运行;开发阶段为 1392 条)。
- **划分**:按整条轨迹随机划分,seed 42:训练 973 / 验证 208 / 测试 210(开发阶段训练集为 974),保存在
  `artifacts/final_run/base/split.json`,所有模型共用。
- **新测试集**:模型冻结后另外采集的 300 个场景(seed 2026,同样的采样范围),不筛选专家是否成功。
- **公开停车时刻**:轮速指令从此之后一直为 0(绝对值 ≤ 1e-3 rad/s)的第一步。只用录制的动作就能得到;
  在抽查的 60 条轨迹上,它和专家内部的 `done` 时刻完全一致。

## 4. 方法概要

所有方案的机械臂都**不直接输出力矩**:网络输出 7×3 矩阵 M,再由标准的 PD + 重力补偿转换成力矩。

```
v      = M(输入) · e                              # e:末端误差,转到底盘坐标系
q_ref ← clip(q_ref + v·dt, q − lead, q + lead)
τ      = Kp (q_ref − q) + Kd (v − qd) + g(q)      # g(q) 由 pinocchio 计算
```

M 和 DLS 逆运动学中的 $J^T(JJ^T+\lambda^2I)^{-1}$ 形状相同(7×3)。网络从示范中学习这个映射,
不需要显式的 Jacobian。

| 方案 | 结构 | 停车信息从哪里来 |
|---|---|---|
| 四网络 | 4 个 MLP:停车点、轮速、停止、机械臂 | 停车点 MLP(从数据学习) |
| GRU v3 | 1 个 GRU,输出轮速、M、停止 logit | 手写的世界方位角特征:sin/cos(atan2(target_y, target_x) − yaw),**与专家 orient 阶段的误差完全相同** |
| **GRU 方案 B** | 停车点 MLP + 1 个 GRU | 停车点 MLP(从数据学习),以连续的相对坐标输入 GRU;同时去掉了会在目标正下方翻转的目标方向角 |

GRU 在停止 logit 连续 N 次为正后锁存:锁存后轮速为 0,机械臂开始伸向目标;锁存前机械臂保持初始姿态。
N 在验证集上开环选择。

**学到的和手写的:**

| | 内容 |
|---|---|
| 从数据学习 | 轮速、停止时机、停车位姿(四网络、方案 B)、机械臂矩阵 M |
| 在验证集上选择 | Kp、Kd、lead、锁存长度 N |
| 手写结构 | 锁存规则;两阶段(先停车再伸臂);参考积分 + PD + 重力补偿(与专家执行层的形式相同,是标准做法) |
| 照搬专家公式 | 仅 GRU v3 的世界方位角特征。站位表在所有当前模型中都没有使用 |
| 机器人模型 | 正运动学、重力项(运行时使用仿真给出的完整底盘姿态) |

## 5. 结果(开发阶段的历史记录;最终结果见第 0 节)

成功判定:沿用模板 `check_stop` 的到达条件,即底盘已完成、末端误差 < 1 cm、t > 8 s,上限 40 s。
**不使用**模板的停滞提前终止(原因见第 6 节)。所有方法(包括专家)使用同一个评估器。

### 5.1 测试集(210 个场景)

| 方案 | 增益 Kp/Kd/lead | 1 cm 成功 | 95% Wilson 区间 | 备注 |
|---|---|---|---|---|
| **GRU 方案 B** | 50/30/0.2 | **206(98.1%)** | 95.2–99.3% | 测试集第三次使用 |
| GRU v3 | 50/30/0.2 | 205(97.6%) | 94.5–99.0% | 第二次使用;增益由 v3 自己的网格选出 |
| GRU v3 | 100/30/0.1 | 204(97.1%) | 93.9–98.7% | 第一次使用;增益来自四网络的网格 |
| 四网络 | 100/30/0.1 | 197(93.8%) | 89.7–96.3% | 第一次使用 |

配对 McNemar 检验:方案 B vs v3,p = 1.0;方案 B vs 四网络,p = 0.035(未做多重比较校正);
v3(50/30/0.2)vs 四网络,p = 0.077。

两次测试之间没有根据测试结果修改模型或增益。方案 B 是在分析了测试集失败之后设计的,
所以它的测试结果应当谨慎看待;两者在验证集上相当(202 vs 203)。

### 5.2 验证集(208 个场景)

| 方案 | 1 cm 成功 |
|---|---|
| 专家底盘 + 专家机械臂 | 208 |
| 专家底盘 + 四网络机械臂(100/30/0.1) | 206 |
| 四网络底盘 + 专家机械臂 | 193 |
| 四网络完整方案(lead 0.15) | 195 |
| GRU v2(无停止输出、无锁存) | 88 |
| GRU v3(100/30/0.1) | 202 |
| GRU v3(50/30/0.2) | 203 |
| GRU 方案 B(50/30/0.2) | 202 |
| 消融:GRU 直接输出力矩 | **2** |

**GRU v3 增益网格**(`gru_v3_grid_val208`):12 组中有 10 组在 199–203 之间。
最差的是 50/10/0.1,只有 184,因为增益太软、跟踪太慢。

### 5.3 专家未成功的 108 个场景

测试集只包含专家成功的场景,存在选择偏差。我们在被排除的 108 个场景上用同一评估器运行了专家和
两个 GRU(增益均为 50/30/0.2)。这些场景没有用于任何训练或选择:

| 场景类型 | 数量 | 专家 | GRU v3 | GRU 方案 B |
|---|---|---|---|---|
| 专家底盘绕停车点打转,导航超时 | 48 | 1 | 47 | 43 |
| 采集脚本判为失败,但按模板标准是成功(误差差 < 0.05 mm,全是低目标) | 51 | 51 | 42 | **23** |
| 到达过早,被模板的停滞规则误判 | 9 | 9 | 7 | 9 |
| 合计 | 108 | 61 | 96 | 75 |

按场景比例加权(普通场景占 1392/1500,用测试集成功率;被排除场景占 108/1500),
未筛选分布上的成功率估计为:**GRU v3 97.0%**,专家 96.9%,GRU 方案 B 96.0%。
方案 B 在贴地目标上明显弱于 v3。

### 5.4 主要失败模式

- **高目标(z > 1.1)**:停车点离目标正下方只有 0.35–0.45 m。v3 的目标方向角在目标正下方会翻转,
  底盘擦过停车点后无法恢复;方案 B 去掉了这个特征,修复了其中大部分。
- **贴地目标(z < 0.25)**:机械臂接近关节限位或奇异位形,学到的 M 误差较大,被 PD 放大后发散。
  两个 GRU 方案都有这个问题。
- **协变量偏移**:974 条训练示范中,专家"越过停车点"的情况出现 0 次,所以模型没有学过如何从中恢复。

## 6. 模板判定的两处问题

1. **停滞判定先于到达判定**(`robotnik_arm_control.py` 中 `check_stop`,第 136–138 行在第 141–142 行之前):
   如果机械臂在 8 s 前就已到达目标并停住,t = 8 s 时会先返回"不可达"。数据集中有 9 例,
   最终误差都 < 4 mm。所以评估时不使用停滞提前终止。
2. **采集脚本的成功标签更严格**(`robotnik_arm_collect.py` 第 320 行,由我们添加):额外要求最后一步之后误差
   < 1 cm,因此多排除了 51 条按模板标准成功的示范(全是低目标)。

## 7. 运行

完整流程用第 0.2 节的四个命令即可。下面是脚本内部的单步命令,便于单独重跑某一步
(路径与 `artifacts/final_run/` 一致;`PY` 换成你自己的 Python,需要 torch、pybullet、pinocchio、simulation_and_control)。

```bash
PY=python
OUT=artifacts/final_run
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# 1. 采集数据(训练用;新测试场景把 --seed 换成 2026)
$PY collect_data.py --n 1500 --seed 42 --workers 6 --output $OUT/collection

# 2. 划分 + 停车点网络(方案 B 使用其中的停车点 MLP)
$PY staged_train.py --source $OUT/collection --output $OUT/base

# 3. 训练 GRU(约 30 分钟,CPU)
$PY staged_gru_train.py --model-dir $OUT/base --stop-head --features bearing --output $OUT/gru_v3_bearing/gru.npz  # v3
$PY staged_gru_train.py --model-dir $OUT/base --stop-head --features park    --output $OUT/gru_B_park/gru.npz      # 方案 B
$PY staged_gru_train.py --model-dir $OUT/base --stop-head --arm-output torque --output $OUT/ablation_torque/gru.npz # 力矩消融

# 4. 在全部 208 个验证场景上选增益(12 组;选出的写在 grid.json 的 "chosen" 中,
#    本次运行 v3 为 100/10/0.1,方案 B 为 200/30/0.2)
$PY staged_arm_grid.py --model-dir $OUT/base --gru-model $OUT/gru_B_park/gru.npz --limit 0 --workers 9 \
    --output $OUT/grid_B_park

# 5. 评估(--split 可选 validation、test、test_unfiltered、excluded;--scenes-dir 直接评估一个场景目录)
$PY staged_evaluate.py --model-dir $OUT/fresh_split --split test_unfiltered --base gru --arm gru \
    --gru-model $OUT/gru_B_park/gru.npz --gains $OUT/grid_B_park/grid.json --workers 18 --output $OUT/my_eval
$PY staged_evaluate.py --model-dir $OUT/fresh_split --split test_unfiltered --base expert --arm expert \
    --workers 18 --output $OUT/my_eval_expert
#    可选:--obs-noise 0.01、--torque-scale 0.8、--push-ee 30、--push-base 20(扰动);
#          --hold-time 2(成功后保持);--gravity zero(去掉重力项);--kp/--kd/--lead(覆盖增益)

# 单个场景(场景 JSON 在 collection 目录或 ood_scenes/ 中)
$PY staged_rollout.py --scene SCENE.json --base gru --arm gru \
    --gru-model $OUT/gru_B_park/gru.npz --kp 200 --kd 30 --lead 0.2 --output OUT_DIR
```

所有脚本都拒绝覆盖已存在的输出目录。在 zsh 中不要把多个参数放进一个变量再展开,zsh 不会自动拆分。

## 8. 局限

- **种子只测了 3 个**(第 0.9 节,验证集 207/207/208),且只换了 GRU 的种子,停车点 MLP 用同一个。
- **开发阶段的测试集被多次使用**,方案 B 的设计受到了测试集失败分析的启发。已通过第 0 节的新测试集
  (冻结后只运行一次)解决;旧测试集上的数字只用于复现核对。
- **不是纯端到端**:机械臂力矩由固定的 PD 层产生;"先停车再伸臂"的两阶段结构和锁存规则是手写的。
  直接输出力矩的消融(验证集 2/208)说明这一层是必要的。PD 层中的重力项来自 pinocchio 模型,但在选出的
  增益下可以去掉(第 0.9 节,207/208 不变);末端位置仍由 pinocchio 正运动学从关节角算出。
- **成功判定只看第一次进入 1 cm**(沿用模板)。增益选择规则因此偏向阻尼小的增益,v3 选中的 Kd = 10
  在保持阶段会来回摆(第 0.9 节)。选择规则应加入保持指标。
- **曝光偏差**:训练只见过专家走过的状态,没有 DAgger 类修正。新测试集 4 个失败中 3 个是从未锁存。
- **鲁棒性实验规模有限**:每个条件 60 个场景、一个扰动强度(推底盘另有校准);分布外每类 40 个场景。
  低目标(z < 0.10)两者都只有约 73%。
- 方案 B 在第 78 轮(上限 80)达到最好,可能尚未完全收敛。
- GRU 的训练轨迹中位 8.6 s、最长 30.4 s;失败的回合会运行到 40 s,超出训练见过的长度。
- 学到的停车点在目标离世界原点很近时误差较大(验证集中有一例偏差 23 cm)。
- GRU 以固定的 100 Hz 更新,改变控制频率需要重新训练。
