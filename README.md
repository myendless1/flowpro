# FlowPRO for Astribot

本仓库是论文 *FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization* 的 Astribot 独立实现，覆盖推理、人工回退接管、偏好对采集、Smooth Interpolation、Wan-VA RPRO 训练和多轮闭环。运行时不依赖 `third_parties/hilserl-astribot` 或 `/media/damoxing/fileset/no4d`。

## 闭环

1. 使用 `wan_va.wan_va_server.VA_Server` 加载本仓库内的 Wan-VA/no4d delta 模型并输出 30-D action；客户端提取 Astribot 16-D delta EEF 命令。
2. `InterventionCollector` 正常执行策略并保留环形回退缓存。
3. 操作者按 **B**：冻结最近 `rollback_horizon` 个策略帧作为 loser，并用实时状态计算 delta EEF 命令回退。
4. 回退完成后按住 **middle trigger**：执行并逐 tick 记录星尘/Quest 专家动作作为 winner。
5. 按 **A**：校验后原子写入 `loser.npz`、`winner.npz`、观测 JSONL 和元数据。未采到接管动作时 A 会拒绝提交。
6. `augment_pair` 对 loser 状态用最近 winner 点、三次 Bézier（位置）、Slerp（姿态）与线性插值（夹爪）生成缺失的 winner chunk；winner 状态构造 identical pair。
7. `rpro_loss` 实现论文 Eq. 3–6。SFT 和正轨迹样本以 identical pair 进入同一目标；第 1 轮 batch 为 current/SFT=80/20，后续轮为 current/history/SFT=70/15/15。winner/loser/current/reference 共享噪声、flow timestep 和条件输入。
8. 每轮训练前冻结当前 transformer 为 reference，训练产物保存到 `checkpoints/last/transformer`；下一轮推理通过 `--transformer-source` 加载该权重。

Quest 和 SDK 命令默认以 100 Hz 更新，策略 waypoint、偏好观测和动作样本以 10 Hz 记录；Wan-VA action chunk 为 32 steps。默认回退 64 个 10 Hz 策略帧，从而在排除最后一个完整 action chunk 后仍能生成负样本。上述频率必须与实际 SFT 数据保持一致。

## 代码对应

- `src/flowpro/collection`：设备无关 B/middle/A 状态机与自动回退。
- `src/astribot_env`：内置 Astribot SDK、RGBD、Quest 坐标转换及接管能力（源自参考工程后独立收录）。其中 `QuestResidualIntervention` 默认按钮 4/5 对应 A/B；接入 collector 时应将 B edge 送入 `InputState.b`、A edge 送入 `InputState.a`，middle 和融合后的绝对 16-D 专家动作一并送入。
- `src/wan_va`：内置 no4d 的模型、配置、数据集、训练和同步推理服务代码。
- `src/flowpro/augmentation`：论文 Smooth Interpolation。
- `src/flowpro/training`：flow loss、真实 preference loader、Wan-VA RPRO trainer 与固定比例 replay mixer。

安装与启动前检查：

```bash
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install lerobot==0.3.3 --no-deps
pip install -e '.[test]'
python scripts/00_validate.py --config configs/flowpro.json --hardware
pytest -q
```

GPU 推理和训练需要提供配置中指定的 Wan2.2/no4d checkpoint 与 LeRobot SFT 数据。Astribot、RGBD、Quest 和远程 Wan-VA 的具体适配器已经实现于 `flowpro.collection.astribot_runtime`。无硬件冒烟采集：

```bash
python -m flowpro.cli.collect --output /tmp/flowpro-pairs --fake --fake-pairs 2 --rollback-horizon 64
```

配置默认使用 8×GPU bf16/DeepSpeed。单卡验证时将 `src/wan_va/configs/accelerate_config.json` 中的 `num_processes` 改为 1，并确保显存足以容纳当前模型、reference 和 optimizer。

## 运行前仍需提供或确认

- `model.base_checkpoint`：必须包含 `vae/`、`tokenizer/`、`text_encoder/` 和 `transformer/`。
- `paths.sft_dataset`：当前配置使用训练就绪的 `centrifuge_multidrop-f1`。它与 `centrifuge_multidrop-f2` 的 531 个 episode、metadata、动作、图像和任务划分一致，但额外包含训练所需的 `empty_emb.pt`、`text_emb/`、`global_text_emb/` 和 latent/cache。原始 `f2` 不能直接交给当前 Wan-VA trainer。
- Python/GPU 依赖：安装 `requirements.txt`，并确认 Accelerate/DeepSpeed 的 GPU 数与机器一致。
- Astribot SDK：通过 `collection.sdk_root` 或 `ASTRIBOT_SDK_ROOT` 配置。
- 任务 prompt：必须与 SFT metadata 完全一致。本数据集支持下列两个 prompt；每轮采集只能选择与现场任务相符的一个：

  - multidrop（当前默认）：`pick up the plate and put it on multidrop`
  - centrifuge：`pick up the plate and put it on centrifuge`

- `collection.init_hdf5`、工作空间上下界、右臂最低高度、碰撞检测和急停：必须在实机试运行前确认。
- GPU 数值训练、checkpoint 回载和真实 Astribot B→回退→middle→A 流程仍需在目标机器上验收；mock 测试不能代替这些验收。

## 数据约束与安全

动作统一为 `[left delta_xyz+relative_wxyz+absolute_gripper, right delta_xyz+relative_wxyz+absolute_gripper]`。机器人适配器按照 Astribot 在线控制示例，以实时 EEF 为参考逐 tick 应用 delta，并包含有限值、单步位移和旋转检查；现场仍必须由 Astribot 控制器提供工作空间、碰撞和急停保护。RGB/状态数组自动写入压缩 NPZ sidecar，JSON 只保存结构和引用。Smooth Interpolation 会排除 loser 的最后一个 action-chunk，避免对接触风险最高的危险尾段做增广。

## 统一配置与脚本

全流程只使用 [configs/flowpro.json](configs/flowpro.json)。相对路径按项目根目录解析，也允许像当前 SFT 数据一样显式配置绝对路径；脚本可以从任意工作目录启动。各阶段会自动使用约定产物：首轮推理读取 `outputs/pretrain`，后续第 N 轮读取 `outputs/rounds/round_(N-1)/offline_rl`。

首次运行必须设置 `collection.prompt`，并通过 `collection.sdk_root` 或环境变量 `ASTRIBOT_SDK_ROOT` 指向 Astribot 官方 SDK。SDK 是硬件驱动依赖，不从参考工程导入。

```bash
# 检查命令与路径，不启动 GPU 或机器人
python scripts/00_validate.py --config configs/flowpro.json --hardware
python scripts/07_run_pipeline.py --config configs/flowpro.json --dry-run

# 分阶段运行
python scripts/01_pretrain.py --config configs/flowpro.json
python scripts/02_inference.py --config configs/flowpro.json --round 1
python scripts/03_collect_preferences.py --config configs/flowpro.json --round 1
python scripts/04_augment_preferences.py --config configs/flowpro.json --round 1
python scripts/05_offline_rl.py --config configs/flowpro.json --round 1

# 自动启动推理服务、等待端口、采集、关闭服务、增广并训练一轮
python scripts/06_run_round.py --config configs/flowpro.json --round 1

# 预训练并运行配置中的全部迭代轮次
python scripts/07_run_pipeline.py --config configs/flowpro.json
```

每一阶段成功后会在 `outputs/manifests` 写入命令、配置和输出路径记录。采集在新增 `collection.target_pairs` 个有效 pair 后自动进入增广和训练；设为 `0` 时持续到 Ctrl-C。

真实机器人第一次运行前必须先以低速、空工作区验证坐标系、四元数顺序、工作空间、碰撞与急停。软件冒烟测试不能替代硬件验收。
