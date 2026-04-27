# Double DQN for CartPole-v1

基于 PyTorch 从零实现的 Double DQN 算法，解决 Gymnasium CartPole-v1 环境。

## 算法简介

Double DQN 在原始 DQN 基础上引入**解耦动作选择与动作评估**机制，使用两个 Q 网络来减少 Q 值过估计问题：

- **当前网络（Policy Net）** — 选择动作
- **目标网络（Target Net）** — 评估动作价值

## 文件说明

| 文件 | 说明 |
|------|------|
| `Main.py` | 完整实现，包含经验回放、Q 网络、训练循环及可视化 |
| `CartPole_Solution.png` | 训练结果曲线图 |

## 环境

- Python 3.8+
- PyTorch
- Gymnasium

```bash
pip install torch gymnasium matplotlib
```

## 运行

```bash
python Main.py
```

## 参考

- [PyTorch DQN Tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461) (Hado van Hasselt et al., 2016)
