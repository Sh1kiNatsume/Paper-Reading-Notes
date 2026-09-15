# World Models

记录世界模型相关论文与学习笔记，重点关注世界模型在手术视频理解、手术流程预测、手术规划和机器人学习中的应用。

## 目录结构

```text
World-Models/
├── Paper-Notes/       # 相关文章阅读记录
│   ├── World-Model-Basics/
│   ├── Video-Prediction-and-Generation/
│   ├── Surgical-Video-Modeling/
│   ├── Robot-Learning-and-Planning/
│   └── Medical-Scene-Challenges/
└── Presentations/     # 自己整理的讲解 PPT 和提纲
```

## 世界模型基本概念

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2018 | [World Models](https://arxiv.org/abs/1803.10122) | 通过视觉观测学习环境的压缩状态和动力学模型 | [Paper](https://arxiv.org/abs/1803.10122)
 | 2019 | [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) | 在潜空间中学习环境动力学并进行规划 | [Paper](https://arxiv.org/abs/1811.04551)
 | 2020 | [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) | 使用潜空间想象进行模型预测和策略学习 | [Paper](https://arxiv.org/abs/1912.01603)
 | 2023 | [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) | 研究世界模型在多种环境中的统一建模和决策 | [Paper](https://arxiv.org/abs/2301.04104)

## 视频预测与生成

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2016 | [Deep Predictive Coding Networks for Video Prediction and Unsupervised Learning](https://arxiv.org/abs/1607.06854) | 使用预测编码学习视频时空表示和未来帧 | [Paper](https://arxiv.org/abs/1607.06854)
 | 2018 | [Stochastic Video Prediction with the Information Bottleneck](https://arxiv.org/abs/1811.05447) | 建模视频未来的不确定性和多种可能结果 | [Paper](https://arxiv.org/abs/1811.05447)
 | 2022 | [Video Diffusion Models](https://arxiv.org/abs/2204.03458) | 将扩散模型应用于视频生成和视频预测 | [Paper](https://arxiv.org/abs/2204.03458)
 | 2024 | [Genie: Generative Interactive Environments](https://arxiv.org/abs/2402.15391) | 从视频学习可交互的生成式环境模型 | [Paper](https://arxiv.org/abs/2402.15391)

## 手术视频建模

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2021 | [Temporal Memory Relation Network for Workflow Recognition from Surgical Video](https://arxiv.org/abs/2103.16327) | 使用记忆关系建模手术视频中的长期流程信息 | [Paper](https://arxiv.org/abs/2103.16327)
 | 2024 | [SANGRIA: Surgical Video Scene Graph Optimization for Surgical Workflow Prediction](https://arxiv.org/abs/2407.20214) | 使用手术场景图预测未来手术工作流 | [Paper](https://arxiv.org/abs/2407.20214)
 | 2025 | [Surgical Video Generation: From Diffusion to World Models](https://arxiv.org/abs/2608.26214) | 总结扩散模型和世界模型在手术视频生成中的应用 | [Paper](https://arxiv.org/abs/2608.26214)
 | 2025 | [SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](https://arxiv.org/abs/2603.13024) | 通过可控视频生成构建手术动作世界模型 | [Paper](https://arxiv.org/abs/2603.13024)
 | 2026 | [SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos](https://arxiv.org/abs/2602.05638) | 面向通用手术视频理解的视频原生基础模型 | [Paper](https://arxiv.org/abs/2602.05638)

## 机器人学习与视觉规划

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2017 | [A Survey of Imitation Learning Techniques for Robot Manipulation](https://arxiv.org/abs/1707.05001) | 介绍从示范中学习机器人操作策略的方法 | [Paper](https://arxiv.org/abs/1707.05001)
 | 2018 | [Deep Visual Foresight for Task-Oriented Robot Learning](https://arxiv.org/abs/1812.01635) | 通过视觉预测模型进行机器人任务规划 | [Paper](https://arxiv.org/abs/1812.01635)
 | 2023 | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) | 使用扩散模型学习视觉运动策略 | [Paper](https://arxiv.org/abs/2303.04137)
 | 2025 | [Cosmos-H-Surgical: Learning Surgical Robot Policies from Videos via World Modeling](https://arxiv.org/abs/2512.23162) | 使用世界模型从手术视频学习机器人策略 | [Paper](https://arxiv.org/abs/2512.23162)

## 医疗场景中的关键问题

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2023 | [EndoNeRF: Neural Radiance Fields for 3D Reconstruction of Deformable Tissues](https://arxiv.org/abs/2206.15255) | 建模内窥镜手术中的三维场景和组织形变 | [Paper](https://arxiv.org/abs/2206.15255)
 | 2023 | [Endoscapes: A Surgical Scene Segmentation and Critical View of Safety Dataset](https://arxiv.org/abs/2312.12429) | 支持手术场景理解和安全视野评估的数据集 | [Paper](https://arxiv.org/abs/2312.12429)
 | 2025 | [SurgBench: A Unified Large-Scale Benchmark for Surgical Video Analysis](https://arxiv.org/abs/2506.07603) | 面向手术视频分析的统一大规模基准 | [Paper](https://arxiv.org/abs/2506.07603)

关注问题：长时序建模、动作稀疏、组织形变、不确定性与安全性，以及跨患者、跨术者和跨中心泛化。

## 推荐阅读顺序

1. [World Models](https://arxiv.org/abs/1803.10122)：理解世界模型、状态表示和环境动力学；
2. [Dreamer](https://arxiv.org/abs/1912.01603)：理解潜空间预测与基于模型的策略学习；
3. [Deep Visual Foresight](https://arxiv.org/abs/1812.01635)：理解视觉预测如何支持机器人规划；
4. [Video Diffusion Models](https://arxiv.org/abs/2204.03458)：理解扩散模型在视频生成中的应用；
5. [SANGRIA](https://arxiv.org/abs/2407.20214)：了解手术场景图和工作流预测；
6. [SAW](https://arxiv.org/abs/2603.13024)：了解手术动作世界模型；
7. [Cosmos-H-Surgical](https://arxiv.org/abs/2512.23162)：了解世界模型在手术机器人策略学习中的应用。

## 方向之间的关系

```text
视频观测
  ↓
状态表示 / 场景理解
  ↓
潜空间动力学 / 视频预测
  ↓
手术流程与未来事件预测
  ↓
视觉规划 / 机器人策略学习
```
