# 强化学习 Atari 游戏项目

这是一个模块化的强化学习项目，实现了 DQN 算法在 Atari 游戏上的训练，包括 Breakout 和 Pong 等经典游戏。

## 项目结构

```
RL/
├── agents/                 # 智能体模块
│   ├── __init__.py
│   ├── random_agent.py    # 随机策略智能体
│   ├── dqn_agent.py       # DQN 智能体
│   └── replay_buffers.py  # 经验回放缓冲区
├── environments/          # 环境模块
│   ├── __init__.py
│   └── atari_env.py      # Atari 环境包装器
├── experiments/          # 实验模块
│   ├── __init__.py
│   ├── breakout_experiment.py  # Breakout 实验
│   └── random_baseline.py      # 随机基线实验
├── utils/                # 工具模块
│   ├── __init__.py
│   ├── config.py         # 配置文件
│   └── training_utils.py # 训练工具函数
├── common/               # 旧版本代码（保留）
├── Breakout.py          # 主入口文件
├── Pong.py              # Pong 游戏入口
└── README.md
```

## 主要特性

- **模块化设计**: 清晰的代码结构，便于维护和扩展
- **多种智能体**: 支持 DQN 和随机策略智能体
- **优先经验回放**: 实现了优先经验回放机制
- **n-step 学习**: 支持 n-step TD 学习
- **实验记录**: 使用 SwanLab 记录训练过程
- **视频录制**: 支持训练后录制游戏视频

## 安装依赖

```bash
pip install torch gymnasium swanlab numpy
```

## 使用方法

### 运行 Breakout 实验

```bash
python Breakout.py
```

这将运行：
1. DQN 训练（5M 帧）
2. 模型评估和视频录制
3. 随机策略基线对比

### 只运行随机基线

```bash
python experiments/random_baseline.py
```

### 只运行 DQN 训练

```python
from experiments.breakout_experiment import run_breakout_experiment
run_breakout_experiment()
```

## 配置说明

主要配置在 `utils/config.py` 中：

- `DEFAULT_CONFIG`: 训练超参数
- `GAME_CONFIGS`: 游戏特定配置
- `SEED`: 随机种子

## 实验结果

训练结果会通过 SwanLab 记录，包括：
- `train/avg_reward`: 训练平均奖励
- `eval/score`: 评估分数
- `train/epsilon`: 探索率
- `train/replay_buffer_size`: 经验池大小

## 扩展新游戏

1. 在 `utils/config.py` 中添加游戏配置
2. 创建新的实验文件（参考 `experiments/breakout_experiment.py`）
3. 运行实验

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License 