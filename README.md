# FlowPRO for Astribot

本仓库是论文 *FlowPRO: Reward-Free Reinforced Fine-Tuning of Flow-Matching VLAs via Proximalized Preference Optimization* 的 Astribot 独立实现，覆盖推理、人工回退接管、偏好对采集、Smooth Interpolation、Wan-VA RPRO 训练和多轮闭环。运行时不依赖 `third_parties/hilserl-astribot` 或 `/media/damoxing/fileset/no4d`。

## 闭环

1. 使用 `wan_va.wan_va_server.VA_Server` 加载 Wan-VA/no4d absolute 或 delta 模型并输出 30-D action；客户端提取 Astribot 16-D 命令，并按照配置解释预测语义。
2. `InterventionCollector` 正常执行策略并保留环形回退缓存。
3. 操作者按 **B**：冻结最近 `rollback_horizon` 个策略帧作为 loser，并反向重放实际执行的 absolute cmd waypoint。
4. 回退完成后按住 **middle trigger**：执行并逐 tick 记录星尘/Quest 专家动作作为 winner。
5. 按 **A**：校验后原子写入 `loser.npz`、`winner.npz`、观测 JSONL 和元数据。未采到接管动作时结束 episode，但不写入 pair。
6. `augment_pair` 对 loser 状态用最近 winner 点、三次 Bézier（位置）、Slerp（姿态）与线性插值（夹爪）生成缺失的 winner chunk；winner 状态构造 identical pair。
7. `rpro_loss` 实现论文 Eq. 3–6。SFT 和正轨迹样本以 identical pair 进入同一目标；第 1 轮 batch 为 current/SFT=80/20，后续轮为 current/history/SFT=70/15/15。winner/loser/current/reference 共享噪声、flow timestep 和条件输入。
8. 每轮训练前冻结当前 transformer 为 reference，训练产物保存到 `checkpoints/last/transformer`；下一轮推理通过 `--transformer-source` 加载该权重。

Quest 和 SDK 命令默认以 100 Hz 更新，策略 waypoint、偏好观测和动作样本以 10 Hz 记录；Wan-VA action chunk 为 32 steps。输入模型的 action history 在两个模式下都使用过去实际下发的 absolute cmd action。默认回退 64 个 10 Hz 策略帧，从而在排除最后一个完整 action chunk 后仍能生成负样本。上述频率必须与实际 SFT 数据保持一致。

## 代码对应

- `src/flowpro/collection`：设备无关 B/middle/A 状态机与自动回退。
- `src/astribot_env`：内置 Astribot SDK、RGBD、Quest 坐标转换及接管能力（源自参考工程后独立收录）。其中 `QuestResidualIntervention` 默认按钮 4/5 对应 A/B；接入 collector 时应将 B edge 送入 `InputState.b`、A edge 送入 `InputState.a`，middle 和融合后的绝对 16-D 专家动作一并送入。
- `src/wan_va`：内置 no4d 的模型、配置、数据集、训练和同步推理服务代码。
- `src/flowpro/augmentation`：论文 Smooth Interpolation。
- `src/flowpro/training`：flow loss、真实 preference loader、Wan-VA RPRO trainer 与固定比例 replay mixer。

## Python 环境

实机采集需要两个 Python 环境。Wan-VA 模型服务和训练使用 Python 3.10；Astribot SDK 的 ROS Noetic 扩展只支持 Python 3.8。二者通过 collector 连接的 `host:port` 通信，因此不需要、也不能在同一解释器中共存。

训练与推理环境（`flowpro`，Python 3.10 / CUDA 12.6）：

```bash
conda activate flowpro
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install lerobot==0.3.3 --no-deps
pip install -e '.[test]'
./scripts/00_validate.sh --hardware
pytest -q
```

机器人采集环境（`astribot`，Python 3.8 / ROS Noetic）：

```bash
conda activate astribot
pip install -r requirements-robot.txt
source /opt/ros/noetic/setup.zsh
source /home/xddex05/repo/astribot_sdk/env.sh
```

每次实际采集前都必须在当前 shell 中执行两个 `source` 命令。它们将 ROS 的 `tf`、消息定义和 Astribot SDK 加入运行环境。不要在 `astribot` 环境中安装 `requirements.txt`，也不要在 `flowpro` 环境中尝试安装 PyPI 的 `tf`；该模块由 ROS Noetic 提供。

真实机器人采集时，先在一个终端启动推理服务：

```bash
cd /home/xddex05/repo/flowpro
./scripts/02_inference.sh --round 1
```

然后在另一个已按上述步骤初始化的 `astribot` 终端启动 collector：

```bash
cd /home/xddex05/repo/flowpro
./scripts/03_collect_preferences.sh --round 1
```

真实机器人采集按 episode 门控执行：

1. 终端提示后按 Quest 右手柄 A，机器人移动到配置的 `init_joint_action` 初始位姿。
2. 整理任务场景，松开 A 后再次按 A，开始执行第一个策略 action chunk。
3. 每个 chunk 完成后进入按键等待：短按 A 执行下一个 chunk，长按 A 至少 2 秒结束本轮，按 B 回退刚完成的 chunk 并进入接管。
4. 接管状态下按住 middle trigger 控制机器人，index trigger 在 `collection.gripper_trigger_threshold` 死区后连续控制夹爪，并按 10Hz 记录实测 state delta 和图像；夹爪每个控制 tick 的最大变化由 `collection.takeover_max_gripper_step` 限制。录到至少一段 correction 后，短按或长按 A 会保存偏好对并结束本轮。按 B 会丢弃当前 loser/winner、不保存也不计入 `collection.target_pairs`，随后保持相同 episode 编号并重新采集这条正负样本。如果回退后发现已经没有必要录制，或任务进入不可能完成的状态，可以不按 middle 直接按 A，本轮会结束但不会保存偏好对、也不会计入 `collection.target_pairs`。
5. 下一轮再次按 A 复位，整理场景后再按 A 开始推理。每个 A 门控都会先等待按键释放，因此上一步的 A 不会误触发下一步。

B 只回退当前 Wan-VA action chunk 中已经执行的动作，不再回退整个 episode。

`03_collect_preferences.sh` automatically loads ROS/Astribot SDK settings,
checks the Quest through ADB, applies `adb reverse tcp:8443 tcp:8443`, and
starts the HTTPS Quest bridge if needed. The bridge runs with `--test`, so it
only publishes controller state; collector remains the sole robot command owner.
Open `https://localhost:8443/quest.html` in the headset browser.

The current host sees the Quest 3S USB vendor as `2833`, but lacks its udev
rule. Once, with the headset connected, run
`sudo ./scripts/setup_quest_adb.sh`, reconnect the USB cable, accept the RSA
debugging prompt in the headset, and verify `adb devices -l` reports `device`.

GPU 推理和训练需要提供配置中指定的 Wan2.2/no4d checkpoint 与 LeRobot SFT 数据。Astribot、RGBD、Quest 和远程 Wan-VA 的具体适配器已经实现于 `flowpro.collection.astribot_runtime`。无硬件冒烟采集：

```bash
python -m flowpro.cli.collect --output /tmp/flowpro-pairs --fake --fake-pairs 2 --rollback-horizon 64
```

配置默认使用 8×GPU bf16/DeepSpeed。单卡验证时将 `src/wan_va/configs/accelerate_config.json` 中的 `num_processes` 改为 1，并确保显存足以容纳当前模型、reference 和 optimizer。

## 运行前仍需提供或确认

- `model.base_checkpoint`：必须包含 `vae/`、`tokenizer/`、`text_encoder/` 和 `transformer/`。
- `paths.sft_dataset`：当前配置使用训练就绪的 `centrifuge_multidrop-f1`。它与 `centrifuge_multidrop-f2` 的 531 个 episode、metadata、动作、图像和任务划分一致，但额外包含训练所需的 `empty_emb.pt`、`text_emb/`、`global_text_emb/` 和 latent/cache。原始 `f2` 不能直接交给当前 Wan-VA trainer。
- Python/GPU 依赖：训练/推理环境安装 `requirements.txt`；实机采集环境安装 `requirements-robot.txt` 并 source ROS Noetic 和 Astribot SDK。确认 Accelerate/DeepSpeed 的 GPU 数与机器一致。
- Astribot SDK：通过 `collection.sdk_root` 或 `ASTRIBOT_SDK_ROOT` 配置。
- 任务 prompt：必须与 SFT metadata 完全一致。本数据集支持下列两个 prompt；每轮采集只能选择与现场任务相符的一个：

  - multidrop（当前默认）：`pick up the plate and put it on multidrop`
  - centrifuge：`pick up the plate and put it on centrifuge`

- `collection.init_joint_action`、工作空间上下界、右臂最低高度、碰撞检测和急停：必须在实机试运行前确认。该关节数组的六组顺序为 torso、left arm、left gripper、right arm、right gripper、head，直接传给 `move_joints_position`，不再依赖 HDF5 首帧。
- GPU 数值训练、checkpoint 回载和真实 Astribot B→回退→middle→A 流程仍需在目标机器上验收；mock 测试不能代替这些验收。

## 数据约束与安全

16-D 命令布局固定为 `[left xyz+wxyz+absolute_gripper, right xyz+wxyz+absolute_gripper]`，但预测语义由 `model.action_representation` 决定：`delta` 模式的 xyz/四元数分别表示位移和相对旋转，`absolute` 模式直接表示 cmd target；夹爪在两个模式下始终是绝对值。机器人适配器会把两种预测都转换成 absolute waypoint 后下发 SDK，并将实际下发的 absolute cmd 写入 history。所有模式都执行有限值、单步位移、旋转和工作空间检查；现场仍必须由 Astribot 控制器提供碰撞和急停保护。RGB/状态数组自动写入压缩 NPZ sidecar，JSON 只保存结构和引用。Smooth Interpolation 会排除 loser 的最后一个 action-chunk，避免对接触风险最高的危险尾段做增广。

## 统一配置与脚本

公共配置位于 [configs/flowpro.json](configs/flowpro.json)，实际双模式入口为 [configs/flowpro.delta.json](configs/flowpro.delta.json) 和 [configs/flowpro.absolute.json](configs/flowpro.absolute.json)。两者通过 `base_config` 继承公共硬件、任务和训练参数，只覆盖 action representation、experiment config、首轮模型权重和输出目录。delta 与 absolute 的 round、manifest 和 pretrain 输出相互隔离，不能在同一个 round 目录混用 preference 数据或 RPRO checkpoint。

相对路径按项目根目录解析，也允许像当前 SFT 数据一样显式配置绝对路径；脚本可以从任意工作目录启动。`paths.pretrain_save_dir` 只作为 SFT/pretrain 阶段的保存根目录；首轮推理和第 1 轮 RPRO reference 读取 `paths.pretrained_transformer_dir`，它可以是直接的 transformer `save_pretrained` 目录，也可以是包含 `transformer/` 或 `checkpoints/last/transformer` 的 checkpoint root。`inference.checkpoint_source` 默认为 `auto`：第 N 轮优先读取当前模式上一轮的 `offline_rl`，不存在时明确告警并回退到 `paths.pretrained_transformer_dir`。设为 `previous_round` 可要求严格使用上一轮 RPRO 权重，设为 `pretrained` 可始终使用预训练权重。

首次运行必须设置 `collection.prompt`，并通过 `collection.sdk_root` 或环境变量 `ASTRIBOT_SDK_ROOT` 指向 Astribot 官方 SDK。SDK 是硬件驱动依赖，不从参考工程导入。

```bash
# Delta mode
./scripts/00_validate.sh --config configs/flowpro.delta.json --hardware
./scripts/02_inference.sh --config configs/flowpro.delta.json --round 1
./scripts/03_collect_preferences.sh --config configs/flowpro.delta.json --round 1

# Absolute mode
./scripts/00_validate.sh --config configs/flowpro.absolute.json --hardware
./scripts/02_inference.sh --config configs/flowpro.absolute.json --round 10
./scripts/03_collect_preferences.sh --config configs/flowpro.absolute.json --round 10
```

同一轮的 `augment`、`offline-rl`、`round` 和 `all` 必须继续传入同一份模式配置。未指定 `--config` 时仍默认使用 `configs/flowpro.json`，其行为与 delta 模式兼容；建议实机流程显式选择上述子配置。

Use `FLOWPRO_CONFIG=/path/to/config.json` for another default, or pass
`--config /path/to/config.json` to one script. `FLOWPRO_TRAIN_PYTHON`,
`FLOWPRO_ROBOT_PYTHON`, `FLOWPRO_QUEST_PYTHON`, and `ADB_SERIAL` override the
corresponding interpreter or device.

`scripts/06_run_round.sh` 和 `scripts/07_run_pipeline.sh` 会用同一个训练解释器自动启动推理与采集，因此仍只适用于离线或模拟流程。实机采集必须按上述分阶段 Bash 脚本运行，保证推理服务和 Python 3.8 Astribot collector 分别处于各自环境中。

每一阶段成功后会在当前模式的 `paths.manifests` 写入命令、配置和输出路径记录。采集在新增 `collection.target_pairs` 个有效 pair 后自动进入增广和训练；设为 `0` 时持续到 Ctrl-C。

真实机器人第一次运行前必须先以低速、空工作区验证坐标系、四元数顺序、工作空间、碰撞与急停。软件冒烟测试不能替代硬件验收。
