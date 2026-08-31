# SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos

> Title: SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos  
> 作者: Jinlin Wu, Felix Holm, Chuxi Chen, An Wang, Yaxin Hu, Xiaofan Ye, Zelin Zang, Miao Xu, Lihua Zhou, Huai Liao, Danny T. M. Chan, Ming Feng, Wai S. Poon, Hongliang Ren, Dong Yi, Nassir Navab, Gaofeng Meng, Jiebo Luo, Hongbin Liu, Zhen Lei  
> 机构: 中国科学院香港人工智能与机器人研究中心；慕尼黑工业大学 Computer Aided Medical Procedures；香港中文大学电子工程系；香港大学深圳医院；中山大学附属第一医院；香港中文大学医学院；北京协和医院；中国科学院自动化研究所；中国科学院大学等  
> Google scholar被引次数: 3  
> 发表: 2026 年 4 月 20 日  
> Venue: arXiv preprint  
> Conference: N/A  
> Year: 2026  
> Link: [arXiv 链接](https://arxiv.org/abs/2602.05638)  
> Code: [GitHub 官方代码仓库](https://github.com/CAIR-HKISI/SurgMotion)  
> Project: [项目主页](https://surgmotion.cares-copilot.com/)

## 1. 背景与问题

### 1.1 论文研究什么？

SurgMotion 研究的是手术视频基础模型，目标是让模型从大规模、多来源的手术视频中学习通用的时空表征，并迁移到多种下游任务。

这些下游任务包括：

- 手术工作流和阶段识别；
- 手术动作识别；
- 器械、动作、目标组织三元组识别；
- 手术技能评估；
- 息肉分割；
- 深度估计。

论文的核心观点是：

> 手术视频基础模型不应该只学习如何重建每个像素，还应该重点学习手术场景中的运动、时空关系以及器械和组织之间的交互。

### 1.2 现有手术视频模型的问题

#### 问题一：像素级重建可能关注低层视觉噪声

VideoMAE 等方法通过遮挡视频区域，再重建被遮挡区域的 RGB 像素。

这种训练目标可能让模型花费大量容量学习：

- 烟雾；
- 反光；
- 液体运动；
- 光照变化；
- 高频纹理；
- 相机运动。

但是这些像素级变化不一定对应核心的手术语义。

模型更应该关注：

- 器械在哪里；
- 器械正在执行什么动作；
- 器械与哪个组织发生交互；
- 当前处于什么手术阶段；
- 不同器械、组织区域之间是什么关系。

#### 问题二：现有方法缺少显式的时空关系建模

手术视频不是独立图像的集合，而是一个连续过程。

例如：

- 器械会沿一定轨迹运动；
- 器械尖端与组织之间会发生交互；
- 不同解剖区域之间存在稳定关系；
- 手术流程具有阶段顺序。

如果只让模型分别预测每个局部区域，可能无法保证多个区域之间的时空关系正确。

#### 问题三：手术视频数据集较为碎片化

已有手术数据集通常只覆盖：

- 一种手术；
- 一种器官；
- 一种机构；
- 一种设备；
- 一个下游任务。

例如，Cholec80 主要用于胆囊切除术的工作流识别，PitVis 主要用于经鼻垂体手术，OphNet 主要用于眼科手术。

这种数据孤岛会限制模型跨手术、跨器官、跨机构和跨设备泛化。

#### 问题四：纹理稀疏场景容易导致表示坍缩

手术视频中存在大面积纹理单一的区域，例如：

- 脂肪组织；
- 平滑器官表面；
- 低纹理背景；
- 光照变化不明显的组织区域。

如果模型长期接触相似的视觉模式，不同特征维度可能输出几乎相同的值，导致：

- 特征通道冗余；
- latent 表示低方差；
- 表示坍缩；
- 下游任务区分能力下降。

### 1.3 论文试图解决的问题

论文要回答的问题可以概括为：

> 能否构建一个视频原生的手术基础模型，使模型从像素级重建转向更加关注运动语义和时空关系，并且能够迁移到不同手术类型、器官和下游任务？

作者提出的整体方案是：

1. 构建 SurgMotion-15M 大规模手术视频数据集；
2. 使用基于 V-JEPA 的 masked latent prediction；
3. 通过运动分数重点关注动态区域；
4. 通过 affinity self-distillation 保持 token 之间的关系；
5. 通过 SFDR 防止特征坍缩；
6. 在多个不同类型的手术视频任务上验证模型。

## 2. 核心方法

### 2.1 SurgMotion 总体结构

SurgMotion 的核心目标是：

> 根据可见的视频区域，预测被 masked 区域在特征空间中的潜在表示，而不是直接预测被遮挡区域的 RGB 像素。

论文模型的整体流程如下：

```text
完整手术视频
      ↓
切分为时空 tube token
      ↓
遮挡一部分 tube
      ↓
可见 tube → Online Encoder → Predictor
                              ↓
                    预测 masked latent

完整视频 → EMA Target Encoder
                    ↓
              生成 target latent

预测 latent 与 target latent 对齐
      ↓
加入运动、关系和特征多样性约束
```

![SurgMotion 模型总体结构](https://raw.githubusercontent.com/CAIR-HKISI/SurgMotion/main/assets/main.png)

图：SurgMotion 项目仓库中的模型与数据概览图。图片来源：[SurgMotion GitHub 仓库](https://github.com/CAIR-HKISI/SurgMotion)。

### 2.2 什么是 latent？

Latent 可以理解为模型内部的潜在表示或隐藏特征。

一张图像原本由大量像素构成，包含：

- 颜色；
- 亮度；
- 纹理；
- 边缘；
- 反光；
- 烟雾；
- 背景。

经过编码器后，图像会被转换成一个特征向量：

```text
原始图像 → 编码器 → latent representation
```

latent 是一组数字，例如：

```text
[0.21, -0.73, 1.14, ..., 0.36]
```

这些数字本身没有直接的人类语义，但整体可能编码：

- 器械类别；
- 器械位置；
- 器械运动；
- 组织结构；
- 器械与组织之间的关系；
- 当前手术阶段；
- 视频片段的时空上下文。

因此：

```text
标签 = 人工定义的明确答案
latent = 模型内部学习到的特征表示
```

### 2.3 为什么不直接预测像素？

像素预测需要模型恢复被遮挡区域的：

- 颜色；
- 纹理；
- 光照；
- 反光；
- 烟雾；
- 液体变化；
- 其他低层视觉细节。

但是，像素变化不一定等于手术语义。

例如，手术画面中可能有一块反光区域发生明显变化：

```text
像素层面：变化很大
手术语义：可能没有重要意义
```

相反，器械尖端轻微地接触组织：

```text
像素层面：变化可能很小
手术语义：可能非常重要
```

因此，SurgMotion 使用 latent prediction：

```text
VideoMAE：
可见视频 → 预测被遮挡像素

SurgMotion：
可见视频 → 预测被遮挡区域的 latent
```

SurgMotion 的目标不是精确生成完整视频，而是学习能够表达手术时空语义的特征。

### 2.4 Online encoder、Predictor 和 EMA target encoder

假设一个视频被分成：

- 可见区域 $x$；
- masked 区域 $y$。

训练过程包括两个分支。

```text
可见区域 x
    ↓
Online Encoder
    ↓
Context features
    ↓
Predictor
    ↓
预测 masked latent

完整视频
    ↓
EMA Target Encoder
    ↓
Target latent
```

定义：

- $E_{\theta}$：online encoder；
- $P_{\phi}$：predictor；
- $\bar{E}_{\theta}$：EMA target encoder；
- $\hat{z}_y$：online 分支预测的 masked latent；
- $z_y^E$：EMA target encoder 产生的目标 latent；
- $\operatorname{sg}$：stop-gradient。

预测结果为：

$$
\hat{z}_y=P_{\phi}(E_{\theta}(x),\Delta_y),
$$

其中 $\Delta_y$ 表示 masked 区域的空间和时间位置。

EMA target encoder 处理完整视频，得到：

$$
z_y^E
=
\bar{E}_{\theta}(X)\big|_y.
$$

这里 $X$ 是完整视频，$\big|_y$ 表示只取 masked 位置的 target feature。

模型要求：

$$
\hat{z}_y\approx z_y^E.
$$

因此，online 分支根据不完整的视频进行预测，EMA target 分支根据完整视频提供目标。

### 2.5 EMA teacher 是什么？

EMA teacher 是 online encoder 的指数移动平均副本。

Online encoder 通过反向传播更新：

$$
\theta_t
=
\theta_{t-1}
-
\eta\nabla_{\theta}\mathcal{L}.
$$

Teacher 不通过当前损失直接反向传播，而是通过指数移动平均更新：

$$
\bar{\theta}_t
=
\mu\bar{\theta}_{t-1}
+
(1-\mu)\theta_t.
$$

其中：

- $\theta_t$：当前 online encoder 参数；
- $\bar{\theta}_t$：当前 teacher 参数；
- $\mu$：动量系数，通常接近 1。

EMA teacher 的作用是：

1. 提供相对稳定的训练目标；
2. 减少当前 mini-batch 噪声；
3. 防止 online 分支和 target 分支同时剧烈变化；
4. 降低表示坍缩风险；
5. 相当于对历史 online encoder 做平滑集成。

需要注意：

> EMA teacher 和 online encoder 在输出上被要求保持一致，但两者的参数并不完全相同。

Online encoder 变化较快，EMA teacher 变化较慢。

### 2.6 Motion-Guided Latent Masked Prediction

不是所有 masked tube 都同样重要。

作者认为，器械、器械尖端、器械与组织交互区域通常具有较强的运动变化，因此先为每个 tube 计算运动分数。

对于 tube $i$，运动分数为：

$$
g_i=
\frac{1}{T-1}
\sum_{t=1}^{T-1}
\left\|
x_{t+1,i}-x_{t,i}
\right\|_1
+
\frac{1}{T}
\sum_{t=1}^{T}
\frac{1}{|\mathcal{N}(i)|}
\sum_{j\in\mathcal{N}(i)}
\left\|
x_{t,i}-x_{t,j}
\right\|_1.
$$

它包含两部分：

1. 当前 tube 在相邻时间帧之间的变化；
2. 当前 tube 与空间邻域之间的差异。

然后在所有 masked tube 中选择运动分数最高的 $K=3$ 个：

$$
I_h
=
\operatorname{TopK}
\left(
\{g_i\}_{i\in I^m},K
\right).
$$

再根据运动分数计算 softmax 权重：

$$
w_i
=
\frac{\exp(\gamma g_i)}
{\sum_{k\in I_h}\exp(\gamma g_k)},
\qquad
\gamma=2.
$$

最终的运动引导预测损失为：

$$
\mathcal{L}_{\mathrm{motion}}
=
\sum_{i\in I_h}
(1+w_i)
\left\|
\hat{z}_i-
\operatorname{sg}(z_i^E)
\right\|_1.
$$

其中：

- $\hat{z}_i$：预测的 masked latent；
- $z_i^E$：EMA target latent；
- $w_i$：运动权重；
- $I_h$：运动最明显的 masked tube 集合。

$1+w_i$ 的作用是：

- 所有被选中的区域都有基础权重 1；
- 运动越强，额外权重越大；
- 不会因为运动较弱而完全忽略某个被选中的区域。

需要注意：

> 运动并不一定等于语义。

烟雾、反光、液体和相机运动也可能产生高运动分数。因此，motion guidance 是一种运动先验，而不是严格的语义标签。

### 2.7 Spatiotemporal Affinity Self-Distillation

#### 2.7.1 为什么需要 affinity loss？

普通 latent prediction 只约束对应 token：

$$
\hat{z}_i\approx z_i^E.
$$

它回答的是：

> 第 $i$ 个区域的特征应该是什么？

但它不一定保证：

> 第 $i$ 个区域与其他区域之间的关系是否正确？

因此，SurgMotion 进一步约束 masked token 之间的关系。

#### 2.7.2 特征归一化

对每个 masked token 进行归一化：

$$
\hat{u}_i
=
\operatorname{norm}(\hat{z}_i),
$$

$$
u_i^E
=
\operatorname{norm}(z_i^E),
$$

其中：

$$
\operatorname{norm}(u)
=
\frac{u}{\max(\|u\|_2,\epsilon)}.
$$

归一化后，模型主要比较特征方向和相似性。

#### 2.7.3 计算 token-to-token 相似度

预测分支中，两个 masked token 的相似度为：

$$
\hat{u}_i^\top\hat{u}_j.
$$

EMA target 分支中，两个 token 的相似度为：

$$
(u_i^E)^\top u_j^E.
$$

如果所有 masked token 被写成矩阵：

$$
\hat{U}
=
[\hat{u}_1,\hat{u}_2,\ldots,\hat{u}_{|I^m|}],
$$

$$
U^E
=
[u_1^E,u_2^E,\ldots,u_{|I^m|}^E],
$$

则可以得到相似度矩阵：

$$
\hat{S}=\hat{U}\hat{U}^{\top},
$$

$$
S^E=U^E(U^E)^{\top}.
$$

矩阵中的元素表示：

```text
S[i,j] = masked token i 与 masked token j 的相似程度
```

#### 2.7.4 转换成 affinity distribution

对于预测分支：

$$
\hat{a}_{i,j}
=
\frac{
\exp(\hat{u}_i^\top\hat{u}_j/\tau)
}{
\sum_{k\in I^m}
\exp(\hat{u}_i^\top\hat{u}_k/\tau)
}.
$$

对于 EMA target 分支：

$$
a^E_{i,j}
=
\frac{
\exp((u_i^E)^\top u_j^E/\tau)
}{
\sum_{k\in I^m}
\exp((u_i^E)^\top u_k^E/\tau)
}.
$$

固定某个 token $i$ 后，$a_{i,:}$ 表示该 token 与所有 masked token 的关系分布。

#### 2.7.5 用 KL 散度对齐关系

最终使用 KL 散度对齐预测关系和目标关系：

$$
\mathcal{L}_{\mathrm{st}}
=
\frac{1}{|I^m|}
\sum_{i\in I^m}
\sum_{j\in I^m}
a^E_{i,j}
\log
\frac{a^E_{i,j}}{\hat{a}_{i,j}}.
$$

这要求：

```text
预测 token 之间的关系
接近
target token 之间的关系
```

因此：

- 普通 latent loss 关注“这个区域是什么”；
- affinity loss 关注“这个区域与其他区域是什么关系”。

作者只在 masked token 内部进行 affinity 蒸馏，避免可见区域和背景 token 在训练中占据过大权重。

#### 2.7.6 Affinity loss 的作用

它可以帮助模型学习：

- 器械区域与组织区域的关系；
- 器械尖端与被操作组织的关系；
- 连续时间中区域关系的稳定性；
- 视角变化下的时空结构；
- 多个局部区域如何共同形成一个手术交互。

例如：

```text
M1：器械尖端
M2：器械接触的组织
M3：远处背景
```

如果 EMA target 认为：

```text
M1 与 M2 高度相关
M1 与 M3 低度相关
```

那么 online 分支不仅要分别预测 M1、M2、M3 的 latent，还要学习相同的关系结构。

### 2.8 Spatiotemporal Feature Diversity Regularization

#### 2.8.1 为什么需要 SFDR？

在纹理单一的手术场景中，模型可能让很多特征维度输出相似的值。

例如：

```text
第 1 维：0.2, 0.2, 0.2, 0.2
第 2 维：0.8, 0.7, 0.8, 0.7
第 3 维：1.1, 0.3, -0.4, 0.9
```

第 1 维几乎是常数，无法提供有效区分信息。

#### 2.8.2 统计每个特征维度的标准差

将一个 batch 中所有预测的 masked latent 汇总为：

$$
\hat{Z}^{m}\in\mathbb{R}^{N\times D}.
$$

其中：

- $N$：所有 masked token 的数量；
- $D$：latent 的特征维度。

对第 $d$ 个特征维度计算标准差：

$$
\sigma_d
=
\operatorname{Std}
\left(
\hat{Z}^{m}_{:,d}
\right).
$$

#### 2.8.3 对低方差维度进行惩罚

论文定义：

$$
\mathcal{L}_{\mathrm{var}}
=
\frac{1}{D}
\sum_{d=1}^{D}
\max(0,\sigma_0-\sigma_d).
$$

其中：

- $\sigma_d$：第 $d$ 个特征维度的标准差；
- $\sigma_0$：最低方差阈值；
- $\max(0,\cdot)$：只有低于阈值时才产生惩罚。

如果：

$$
\sigma_d<\sigma_0,
$$

该维度会受到惩罚。

如果：

$$
\sigma_d\geq\sigma_0,
$$

该维度在 SFDR 中不再受到额外惩罚。

#### 2.8.4 对 SFDR 的正确理解

SFDR 不是：

```text
低方差维度 → 少关注
高方差维度 → 多关注
```

更准确的是：

```text
低方差维度
    ↓
产生惩罚
    ↓
反向传播
    ↓
推动该维度重新产生与输入有关的变化
```

因此，SFDR 不会直接删除低方差维度，也不会简单把它的权重调小。

它要求模型：

> 不要让某些特征通道退化成几乎恒定的常数。

第 3 维变化较大，只表示它已经满足方差要求，SFDR 不再额外惩罚它；不能简单地说模型一定会更多关注第 3 维。

#### 2.8.5 SFDR 与 affinity loss 的区别

| 方法 | 统计对象 | 约束内容 |
|---|---|---|
| Affinity self-distillation | 不同 masked token 之间 | token 与 token 的关系 |
| SFDR | 同一特征维度在 batch 中 | 特征维度的变化程度 |

可以记为：

```text
Affinity：横向关系结构
SFDR：纵向特征维度
```

### 2.9 总体训练目标

SurgMotion 的最终训练目标为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{motion}}
+
\lambda_{\mathrm{st}}\mathcal{L}_{\mathrm{st}}
+
\lambda_{\mathrm{var}}\mathcal{L}_{\mathrm{var}}.
$$

其中：

- $\mathcal{L}_{\mathrm{motion}}$：运动引导的 masked latent prediction；
- $\mathcal{L}_{\mathrm{st}}$：时空 affinity 自蒸馏；
- $\mathcal{L}_{\mathrm{var}}$：特征多样性正则化；
- $\lambda_{\mathrm{st}}=0.1$；
- $\lambda_{\mathrm{var}}=0.3$。

三个模块的分工为：

| 模块 | 重点 | 解决的问题 |
|---|---|---|
| Motion-guided prediction | 动态区域 | 让模型关注器械和器械—组织交互 |
| Affinity self-distillation | token 间关系 | 保持时空结构稳定 |
| SFDR | 特征维度多样性 | 防止表示坍缩和通道冗余 |

![SurgMotion 框架图](https://raw.githubusercontent.com/CAIR-HKISI/SurgMotion/main/assets/framework.png)

图：SurgMotion 官方代码仓库中的框架示意图。图片来源：[SurgMotion GitHub 仓库](https://github.com/CAIR-HKISI/SurgMotion)。

## 3. 创新点

- 将手术视频预训练目标从像素空间重建转向特征空间中的 latent prediction。
- 基于 V-JEPA 的 masked feature prediction 思想，构建面向手术视频的基础模型。
- 提出 Motion-Guided Latent Masked Prediction，利用运动分数突出动态手术区域。
- 使用 Top-$K$ 运动 tube 和 softmax 权重增强对高运动区域的训练。
- 提出 Spatiotemporal Affinity Self-Distillation，使模型学习 masked token 之间的时空关系。
- 使用 KL 散度对齐 online 分支和 EMA target 分支的 affinity distribution。
- 提出 SFDR，对低方差特征维度施加惩罚，减少表示坍缩和通道冗余。
- 构建 SurgMotion-15M 多来源、多器官、多手术类型的预训练数据集。
- 采用专业方向和数据集级别的均衡采样，避免某个大数据集主导训练。
- 使用 16 帧到 64 帧的两阶段预训练策略，逐步扩大时间上下文。
- 在工作流识别、动作识别、动作三元组识别、技能评估、息肉分割和深度估计等任务上进行验证。
- 证明同一个预训练编码器可以迁移到分类、关系识别、分割和深度估计等不同任务。

## 4. 实验 & 结果

### 4.1 SurgMotion-15M 数据集

SurgMotion-15M 的构建思路是：

```text
多个公开手术数据集
        +
机构内部手术视频
        ↓
统一整理为大规模手术视频预训练数据
```

论文报告的数据规模包括：

- 50 个视频来源；
- 约 3,658 小时视频；
- Figure 1 中报告约 45,757 个视频；
- 13 个解剖区域；
- 12 类主要专业方向；
- 70 多种解剖区域；
- 100 多种手术操作。

主要覆盖：

- 神经外科；
- 普通外科；
- 眼科；
- 胃部手术；
- 腹腔镜胆囊切除；
- 结肠镜；
- 妇科；
- 泌尿外科；
- 肝脏手术；
- 技能训练；
- 支气管镜；
- 乳腺手术。

主要数据来源包括：

- PitVis；
- AVOS；
- OphNet；
- CATARACTS；
- MultiBypass140；
- CoPESD；
- Cholec80；
- M2CAI-2016；
- EndoFM；
- AutoLaparo；
- PSI-AVA；
- PmLR50；
- JIGSAWS；
- AIxSuture；
- 机构内部的支气管镜和乳腺手术数据。

### 4.2 数据采样与训练

预训练时：

- 视频按照 1 fps 处理；
- 每帧短边缩放到 256 像素；
- 随机裁剪为 $224\times224$；
- tube-level mask ratio 为 0.75–0.95；
- Stage 1 使用 16 帧，训练 200 epochs；
- Stage 2 使用 64 帧，继续训练 80 epochs；
- 使用 AdamW 优化器；
- EMA momentum 为 0.99925；
- 使用 4 张 NVIDIA H800 GPU；
- Stage 1 global batch size 为 256；
- Stage 2 global batch size 为 64。

为了平衡不同专业和数据集的影响，作者采用：

$$
w(x)\propto
\frac{1}{N_d|D_s|},
$$

其中：

- $N_d$：数据集 $d$ 的规模；
- $D_s$：专业方向 $s$ 下的数据集数量。

该策略可以降低大数据集中单个样本的采样概率，同时提高小数据集被看到的机会。

### 4.3 手术工作流识别

在腹腔镜数据集上：

| 数据集 | Accuracy | F1 | Jaccard |
|---|---:|---:|---:|
| Cholec80 | 91.05 | 84.17 | 77.95 |
| AutoLaparo | 86.37 | 78.73 | 69.86 |
| M2CAI-2016 | 89.45 | 82.91 | 75.42 |

在开放手术、经鼻手术和神经外科任务上：

| 数据集 | Accuracy | F1 | Jaccard |
|---|---:|---:|---:|
| EgoSurgery | 75.57 | 50.72 | 42.95 |
| PitVis | 86.52 | 75.53 | 65.51 |
| Atlas-Neurosurgical | 83.48 | 65.26 | 59.35 |

论文报告：

- EgoSurgery 上比最佳手术基础模型高 14.6 个百分点 F1；
- PitVis 上比 GastroNet 高 10.3 个百分点 F1；
- Atlas-Neurosurgical 上达到 65.26% F1。

在眼科和肝脏手术数据集上：

| 数据集 | Accuracy | F1 | Jaccard |
|---|---:|---:|---:|
| OphNet | 73.04 | 51.22 | 43.83 |
| PmLR50 | 91.91 | 85.65 | 78.56 |

这些结果支持模型在不同手术类型和不同成像场景下具有一定迁移能力。

### 4.4 动作识别

| 数据集 | Accuracy | F1 |
|---|---:|---:|
| SurgicalActions160 | 75.63 | 72.62 |
| AVOS | 70.52 | 66.32 |
| PolypDiag | 98.81 | 98.34 |

在 AVOS 开放手术数据集上，SurgMotion 的 F1 为 66.32%，高于 DINOv3-H 的 45.12%，提升 21.2 个百分点。

### 4.5 动作三元组识别

CholecT50 将动作定义为：

$$
\langle\text{Instrument},\text{Verb},\text{Target}\rangle.
$$

例如：

```text
〈grasper, retract, gallbladder〉
抓钳、牵拉、胆囊

〈clipper, clip, cystic-duct〉
夹钳、夹闭、胆管

〈scissors, cut, cystic-duct〉
剪刀、切割、胆管
```

SurgMotion 的结果为：

| 指标 | 结果 |
|---|---:|
| AP-I，器械 | 91.55 |
| AP-V，动作 | 57.72 |
| AP-T，目标 | 48.18 |
| AP-IV | 40.39 |
| AP-IT | 43.47 |
| AP-IVT | 39.54 |


### 4.6 CholecT50 为什么比阶段识别更难？

阶段识别回答：

```text
当前处于什么手术阶段？
```

例如：

```text
clipping and cutting
```

但是一个阶段内可能同时发生多个不同动作：

```text
抓钳牵拉胆囊
夹钳夹闭胆管
剪刀切割胆管
```

三元组识别需要回答：

```text
哪个器械？
执行什么动作？
作用于哪个目标组织？
```

同一个器械可能执行不同动作：

```text
〈grasper, grasp, gallbladder〉
〈grasper, retract, gallbladder〉
〈grasper, dissect, gallbladder〉
```

同一个目标也可能同时被多个器械操作。

因此，该任务本质上不只是三个独立分类问题，而是器械、动作和目标之间的三方关系关联问题。

### 4.7 技能评估

在 JIGSAWS 上：

| 指标 | SurgMotion |
|---|---:|
| MAE | 2.649 |
| Spearman correlation | 0.770 |

与 GastroNet 相比：

- MAE 从 3.596 降低到 2.649；
- Spearman 从 0.500 提升到 0.770。

这表明 SurgMotion 学到的表征可能不仅包含“做了什么动作”，还包含一定程度的动作质量和流畅性信息。

### 4.8 息肉分割

| 数据集 | 类型 | Dice |
|---|---|---:|
| CVC-ClinicDB | In-domain | 0.8694 |
| Kvasir | In-domain | 0.8919 |
| CVC-300 | Out-of-distribution | 0.9112 |
| CVC-ColonDB | Out-of-distribution | 0.7985 |
| ETIS-LaribPolypDB | Out-of-distribution | 0.7798 |

在 OOD 数据集上，SurgMotion 取得了较强结果，说明预训练表征可能具有一定的跨数据集迁移能力。

### 4.9 深度估计

在 C3VD 结肠镜深度估计任务上：

| 指标 | SurgMotion |
|---|---:|
| RMSE | 1.88 |
| AbsRel | 0.051 |
| SqRel | 0.13 |
| $\delta<1.1$ | 0.86 |

这说明视频预训练得到的时空表示可能包含有利于空间结构和深度预测的信息。

### 4.10 论文结果的总体理解

SurgMotion 的实验设计覆盖了多个层次：

```text
阶段识别
    ↓
动作识别
    ↓
器械-动作-目标关系
    ↓
技能质量评估
    ↓
像素级分割
    ↓
深度结构预测
```

因此，作者试图证明：

> SurgMotion 学到的不是只适用于某个分类任务的特征，而是能够迁移到多个粒度和多个输出形式的通用视频表征。

### 4.11 与相关方法的区别

#### SurgMotion 与 VideoMAE

| 项目 | VideoMAE | SurgMotion |
|---|---|---|
| 预测目标 | 被遮挡区域的 RGB 像素 | 被遮挡区域的 latent |
| 训练方式 | 像素空间重建 | 特征空间预测 |
| Masking | 高比例 tube masking | 高比例 masking，并加入运动引导 |
| 主要关注 | 视频时空结构和像素重建 | 手术运动语义和时空关系 |
| 领域 | 通用视频 | 手术视频 |

#### SurgMotion 与 V-JEPA

| 项目 | V-JEPA | SurgMotion |
|---|---|---|
| 核心思想 | masked latent prediction | masked latent prediction |
| Teacher | EMA target encoder | EMA target encoder |
| 预测损失 | latent $L_1$ | 运动加权 latent $L_1$ |
| 领域 | 通用视频 | 手术视频 |
| 额外机制 | 基础 feature prediction | motion guidance、affinity、SFDR |

#### SurgMotion 与 Endo-FM

| 项目 | Endo-FM | SurgMotion |
|---|---|---|
| 输入构造 | global/local views | masked/unmasked tubes |
| 主要目标 | 跨视角特征匹配 | masked latent prediction |
| 运动建模 | 不同帧率之间进行匹配 | 对动态区域增加预测权重 |
| Teacher-student | 使用 EMA teacher | 使用 EMA teacher |
| 下游任务 | 分类、分割、检测 | 工作流、动作、技能、分割、深度 |

#### SurgMotion 与 SurgVLP

SurgVLP 的核心是视频—文本对齐：

```text
视频 → 视频编码器 → 视频 latent
文本 → 文本编码器 → 文本 latent
                ↓
          对比学习对齐
```

SurgMotion 的核心是：

```text
可见手术视频 → 预测 masked latent
```

两者都使用 latent representation，但训练信号不同：

- SurgVLP：让匹配的视频和文本靠近；
- SurgMotion：让预测的 masked latent 接近 EMA target latent。

#### SurgMotion 与 SurgVLM

SurgVLM 建立在 Qwen2.5-VL 上，使用手术图像或视频、问题和结构化标签进行 instruction tuning。

```text
手术图像或视频 + 问题
          ↓
视觉语言模型
          ↓
生成文本答案或结构化结果
```

SurgMotion 主要是视频表征学习模型，SurgVLM 主要是面向多任务问答和推理的视觉语言大模型。

#### SurgMotion 与 SurgicalSAM / SurgicalSAM-2

SurgicalSAM 和 SurgicalSAM-2 主要解决手术器械分割问题。

SurgicalSAM-2 的核心是 Efficient Frame Pruning：

```text
当前帧
  ↓
与历史帧计算余弦相似度
  ↓
删除冗余历史帧
  ↓
保留更有信息的历史帧
  ↓
进行视频分割
```

它重点关注：

- 分割准确率；
- FPS；
- 显存占用；
- memory bank 管理。

SurgMotion 重点关注：

- 视频自监督预训练；
- latent representation；
- 运动语义；
- 时空关系；
- 跨任务迁移。

## 5. 个人思考 & 待读问题

### 5.1 论文最值得学习的地方

这篇论文的逻辑是比较清晰的：

```text
手术视频中存在大量低层视觉噪声
        ↓
像素级重建不一定适合
        ↓
转向 latent prediction
        ↓
手术语义与局部运动相关
        ↓
加入 motion guidance
        ↓
手术场景具有复杂时空关系
        ↓
加入 affinity self-distillation
        ↓
纹理稀疏区域可能造成表示坍缩
        ↓
加入 SFDR
        ↓
使用多来源数据提升跨任务和跨手术泛化
```

论文不是先设计复杂模块，再寻找应用场景，而是把不同问题分别映射到具体方法：

| 问题 | 方法 |
|---|---|
| 像素噪声 | latent prediction |
| 动态语义不突出 | motion-guided prediction |
| token 关系不稳定 | affinity self-distillation |
| 特征坍缩 | SFDR |
| 数据碎片化 | SurgMotion-15M |

### 5.2 性能提升是否主要来自算法？

这是最重要的待研究问题之一。

SurgMotion 使用了大规模、多来源的预训练数据，而部分对比模型的训练数据规模可能较小。因此最终性能提升可能同时来自：

- 更大的预训练数据；
- 更丰富的数据分布；
- 更长的视频训练；
- 更强的模型结构；
- motion-guided prediction；
- affinity distillation；
- SFDR。

如果没有严格控制数据规模和训练预算，就不能简单地认为所有提升都来自三个新模块。

建议进一步进行：

```text
相同数据规模
相同模型规模
相同训练步数
相同下游评估协议

VideoMAE
V-JEPA
SurgMotion
```

之间的公平比较。

### 5.3 三个模块是否都必要？

需要进行完整的消融实验：

| 版本 | Motion | Affinity | SFDR |
|---|---:|---:|---:|
| Baseline | 否 | 否 | 否 |
| Model A | 是 | 否 | 否 |
| Model B | 否 | 是 | 否 |
| Model C | 否 | 否 | 是 |
| Model D | 是 | 是 | 否 |
| Model E | 是 | 否 | 是 |
| Full model | 是 | 是 | 是 |

还可以测试：

- 不同的 motion top-$K$；
- 不同的 mask ratio；
- 不同的 $\lambda_{\mathrm{st}}$；
- 不同的 $\lambda_{\mathrm{var}}$；
- 是否使用 EMA teacher；
- 16 帧和 64 帧的差异；
- 是否使用 affinity loss；
- 是否使用 SFDR。

### 5.4 motion score 是否真正对应手术语义？

motion score 根据像素变化计算，因此可能误选：

- 烟雾；
- 反光；
- 液体；
- 相机移动；
- 光照变化。

可以进一步分析运动最高区域是否集中在：

- 器械；
- 器械尖端；
- 器械—组织交互区域；
- 组织形变区域。

可能的改进方向包括：

- 使用器械检测或分割辅助运动区域选择；
- 结合光流与语义分割；
- 使用可学习的运动显著性模块；
- 比较 motion score 与人工标注交互区域的重合率。

### 5.5 SFDR 是否可能引入无意义变化？

SFDR 会惩罚低方差特征维度，推动它们产生更多变化。

但如果约束过强，模型可能通过产生噪声来满足方差要求。

因此需要进一步判断：

```text
特征维度变化增加
是否真的对应手术语义？
```

而不是只产生随机变化。

理想情况下：

- motion loss 让特征符合动态区域；
- affinity loss 让 token 关系保持正确；
- SFDR 让特征维度不要坍缩。

### 5.6 SurgMotion-15M 数据集是否完全可复现？

论文明确报告了数据规模和覆盖范围，但以下内容仍需要核对：

- 数据如何去重；
- 不同公开数据集之间是否存在重复视频；
- 预训练数据和测试数据是否存在病例重叠；
- 私有数据如何获得授权；
- 私有数据如何脱敏；
- 低质量视频如何清洗；
- 不同专业方向的真实采样比例；
- 各机构和设备的具体占比。

此外，论文中的部分数据统计存在需要核查的地方：

- Figure 1 中神经外科数据的小时数显示约 22,288.7 小时；
- Table 1 中神经外科数据为约 2,860.85 小时；
- Figure 1、Table 1 和摘要中的视频数量及小时数存在不完全一致；
- 论文和官方代码仓库说明 SurgMotion-15M 包含 15M frames，但论文正文的图表对“15M”的统计单位说明不够统一。

正式汇报时，建议写成：

> 论文和官方代码仓库将数据集称为 SurgMotion-15M，并报告其包含约 15M 帧、3,658 小时、50 个来源和 13 个以上解剖区域；但部分图表中的时长统计存在差异，需要进一步核对原始数据。

### 5.7 是否真正实现了跨机构泛化？

论文强调跨手术和跨领域泛化，但还需要进一步确认训练集和测试集是否在机构、设备和病例层面完全隔离。

更有说服力的实验设置应该是：

```text
医院 A、B 训练
医院 C 测试
```

或者：

```text
一种手术训练
另一种手术测试
```

还可以进行：

- 跨医院测试；
- 跨设备测试；
- 跨分辨率测试；
- 跨成像方式测试；
- 跨手术类型测试。

### 5.8 如果自己复现，应该怎么做？

建议从简单到复杂逐步实现。

#### 第一步：实现基础 masked latent prediction

```text
视频
  ↓
随机 mask tube
  ↓
Online encoder + predictor
  ↓
EMA target encoder
  ↓
masked latent L1 loss
```

#### 第二步：加入 motion guidance

比较：

```text
普通 masked latent prediction
vs.
运动加权 masked latent prediction
```

#### 第三步：加入 affinity loss

比较：

```text
只对齐单个 token
vs.
同时对齐 token-to-token affinity
```

#### 第四步：加入 SFDR

观察：

- latent 每个维度的标准差；
- 特征协方差；
- 是否出现通道坍缩；
- 下游任务性能是否提升。

#### 第五步：控制实验条件

保证：

- 相同预训练数据；
- 相同模型规模；
- 相同训练步数；
- 相同 mask ratio；
- 相同下游任务；
- 相同 probing 设置。

### 5.9 SurgMotion 在手术视频研究中的位置

手术视频理解大致经历了以下发展：

```text
早期单帧器械识别和分割
        ↓
手术阶段和工作流识别
        ↓
CNN + LSTM 等时序建模
        ↓
多任务学习
        ↓
更细粒度的器械、动作和组织识别
        ↓
器械-动作-目标三元组
        ↓
场景理解和关系建模
        ↓
VideoMAE 等视频自监督预训练
        ↓
V-JEPA 等 latent prediction
        ↓
Endo-FM 等内窥镜视频基础模型
        ↓
SurgVLP / SurgVLM 等视觉语言模型
        ↓
SurgMotion 等面向手术视频的基础模型
```

SurgMotion 的位置可以概括为：

> 它将 V-JEPA 的 latent prediction 思想与手术视频中的运动先验、时空关系和跨任务迁移结合起来，尝试构建更加通用的手术视频基础模型。

### 5.10 最终总结

SurgMotion 的核心贡献可以总结为：

> SurgMotion 基于 V-JEPA，将手术视频 masked modeling 的目标从像素空间转移到 latent 特征空间，并通过运动引导、时空 affinity 自蒸馏和特征多样性正则化，使模型更加关注器械运动、器械—组织交互和跨区域时空关系。作者同时构建了大规模、多来源的 SurgMotion-15M 数据集，并通过多种手术视频下游任务验证其表征的迁移能力。

理解：

```text
VideoMAE：看缺块，补像素
V-JEPA：看缺块，补 latent
Endo-FM：不同视角下保持视频特征一致
SurgMotion：重点预测动态区域的 latent，并保持时空关系和特征多样性
```
# SurgMotion 关键损失函数计算示例

> 以下矩阵均为人为构造的示意数据，用于说明计算过程，不代表论文中的真实训练输出。

## 一、时空 Affinity Self-Distillation 计算示例

假设有 3 个 masked tokens，每个 token 的 latent 维度为 4：

- Token 1：器械尖端；
- Token 2：被接触组织；
- Token 3：无关背景。

### 1. 教师 latent 矩阵

$$
Z^E=
\begin{bmatrix}
0.90 & 0.40 & 0 & 0\\
0.88 & 0.42 & 0.22 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}.
$$

其中：

$$
z_1^E=(0.90,0.40,0,0),
$$

$$
z_2^E=(0.88,0.42,0.22,0),
$$

$$
z_3^E=(0,0,0,1).
$$

### 2. 教师 latent 的 L2 归一化

论文采用：

$$
u_i^E=\frac{z_i^E}{\max(\|z_i^E\|_2,\epsilon)}.
$$

各向量的范数为：

$$
\|z_1^E\|_2=\sqrt{0.90^2+0.40^2}=\sqrt{0.97}\approx 0.9849,
$$

$$
\|z_2^E\|_2=\sqrt{0.88^2+0.42^2+0.22^2}=\sqrt{0.9992}\approx 0.9996,
$$

$$
\|z_3^E\|_2=1.
$$

因此，归一化后的教师矩阵为：

$$
U^E\approx\begin{bmatrix}
0.9138 & 0.4061 & 0 & 0\\
0.8804 & 0.4202 & 0.2201 & 0\\
0 & 0 & 0 & 1\end{bmatrix}.
$$

### 3. 教师两两 token 相似度矩阵

归一化后进行矩阵乘法：

$$
S^E=U^E(U^E)^\top.
$$

由于每个 token 已经经过 L2 归一化，矩阵元素就是余弦相似度。

例如：

$$
S^E_{1,2}=u_1^E\cdot u_2^E\approx0.9138\times0.8804+0.4061\times0.4202\approx 0.9751.
$$

教师中 Token 2 与 Token 3 的相似度为：

$$
S^E_{2,3}=u_2^E\cdot u_3^E=0.
$$

因此：

$$
S^E\approx\begin{bmatrix}
1 & 0.9751 & 0\\
0.9751 & 1 & 0\\
0 & 0 & 1\end{bmatrix}.
$$

这表示：

- Token 1 与 Token 2 具有很强的关系；
- Token 2 与 Token 3 基本没有关系；
- Token 3 主要与自身相关。

### 4. 教师 affinity distribution

论文使用温度系数：

$$
\tau=0.1.
$$

对相似度矩阵逐行执行带温度的 softmax：

$$
a^E_{i,j}=\frac{\exp(S^E_{i,j}/\tau)}{\sum_k\exp(S^E_{i,k}/\tau)}.
$$

教师相似度除以温度后为：

$$
\frac{S^E}{\tau}\approx\begin{bmatrix}10 & 9.751 & 0\\
9.751 & 10 & 0\\
0 & 0 & 10\end{bmatrix}.
$$

逐行 softmax 后：

$$
A^E\approx\begin{bmatrix}
0.5601 & 0.4399 & 0.000025\\
0.4399 & 0.5601 & 0.000025\\
0.000045 & 0.000045 & 0.999909\end{bmatrix}.
$$

第一行表示教师认为 Token 1：

- 与自身的关联权重约为 $56.01\%$；
- 与 Token 2 的关联权重约为 $43.99\%$；
- 与 Token 3 几乎没有关系。

---

### 5. 学生 latent 矩阵

假设学生输出为：

$$
\hat Z=\begin{bmatrix}0.90 & 0.40 & 0 & 0\\
0.78 & 0.48 & 0.22 & 0.32\\
0 & 0 & 0 & 1\end{bmatrix}.
$$

学生 Token 2 在第 4 个特征维度上出现了 $0.32$。该维度主要对应背景 Token 3，因此学生可能错误地将背景信息混入了 Token 2。

### 6. 学生 latent 的 L2 归一化

学生 Token 1 和 Token 3 的归一化结果为：

$$
\hat u_1\approx(0.9138,0.4061,0,0),
$$

$$
\hat u_3=(0,0,0,1).
$$

Token 2 的范数为：

$$
\|\hat z_2\|_2=\sqrt{0.78^2+0.48^2+0.22^2+0.32^2}=\sqrt{0.9896}\approx0.9948.
$$

因此：

$$
\hat u_2\approx(0.7841,0.4825,0.2211,0.3217).
$$

归一化后的学生矩阵为：

$$
\hat U\approx\begin{bmatrix}
0.9138 & 0.4061 & 0 & 0\\
0.7841 & 0.4825 & 0.2211 & 0.3217\\
0 & 0 & 0 & 1\end{bmatrix}.
$$

### 7. 学生两两 token 相似度矩阵

学生相似度矩阵为：

$$
\hat S=\hat U\hat U^\top.
$$

Token 1 与 Token 2 的相似度为：

$$
\hat S_{1,2}=\hat u_1\cdot\hat u_2\approx0.9125.
$$

Token 2 与 Token 3 的相似度为：

$$
\hat S_{2,3}=\hat u_2\cdot\hat u_3=0.3217.
$$

因此：

$$
\hat S\approx\begin{bmatrix}
1 & 0.9125 & 0\\
0.9125 & 1 & 0.3217\\
0 & 0.3217 & 1\end{bmatrix}.
$$

与教师矩阵相比：

$$
S^E\approx\begin{bmatrix}1 & 0.9751 & 0\\
0.9751 & 1 & 0\\
0 & 0 & 1\end{bmatrix},
$$

$$
\hat S\approx\begin{bmatrix}1 & 0.9125 & 0\\
0.9125 & 1 & 0.3217\\
0 & 0.3217 & 1\end{bmatrix}.
$$

关键变化为：

- 正确关系 Token 1–Token 2：$0.9751\rightarrow0.9125$；
- 错误关系 Token 2–Token 3：$0\rightarrow0.3217$。

### 8. 学生 affinity distribution

学生相似度除以温度后为：

$$
\frac{\hat S}{\tau}\approx\begin{bmatrix}10 & 9.125 & 0\\
9.125 & 10 & 3.217\\
0 & 3.217 & 10\end{bmatrix}.
$$

逐行 softmax 后：

$$
\hat A\approx\begin{bmatrix}0.7058 & 0.2942 & 0.000032\\
0.2940 & 0.7052 & 0.0008\\
0.000045 & 0.0011 & 0.9988\end{bmatrix}.
$$

学生第二行相较教师第二行：

$$
a_2^E\approx(0.4399,0.5601,0.000025),
$$

$$
\hat a_2\approx(0.2940,0.7052,0.0008).
$$

这说明学生：

- 降低了 Token 2 与 Token 1 的关联；
- 增加了 Token 2 与背景 Token 3 的错误关联；
- 更倾向于让 Token 2 只关注自身。

### 9. KL 散度

论文使用：

$$
\mathcal{L}_{\mathrm{st}}=\frac{1}{|I^m|}\sum_{i\in I^m}\sum_{j\in I^m}a^E_{i,j}\log\frac{a^E_{i,j}}{\hat a_{i,j}}.
$$

本例中共有 3 个 masked tokens，因此：

$$
\mathcal{L}_{\mathrm{st}}=\frac{1}{3}\left[D_{\mathrm{KL}}(a_1^E\|\hat a_1)+D_{\mathrm{KL}}(a_2^E\|\hat a_2)+D_{\mathrm{KL}}(a_3^E\|\hat a_3)\right].
$$

各行 KL 散度约为：

$$
D_{\mathrm{KL}}(a_1^E\|\hat a_1)\approx0.047,
$$

$$
D_{\mathrm{KL}}(a_2^E\|\hat a_2)\approx0.047,
$$

$$
D_{\mathrm{KL}}(a_3^E\|\hat a_3)\approx0.0009.
$$

因此：

$$
\mathcal{L}_{\mathrm{st}}\approx\frac{0.047+0.047+0.0009}{3}\approx\boxed{0.032}.
$$

### 10. Affinity loss 的含义

这个例子说明，Affinity loss 不只要求：

$$
\hat z_i\approx z_i^E,
$$

还要求：

$$
\text{学生 token 间的关系}\approx\text{教师 token 间的关系}.
$$

它可以同时惩罚：

1. 学生削弱了本应存在的 Token 1–Token 2 关系；
2. 学生错误建立了 Token 2–Token 3 的背景关系。

---

## 二、SFDR 中 $\mathcal{L}_{\mathrm{var}}$ 的计算示例

### 1. 假设的 latent 矩阵

假设一个 batch 中有 4 个 masked tokens，每个 token 有 3 个特征维度：

$$
Z=\begin{bmatrix}0.2 & 0.8 & 1.1\\
0.2 & 0.7 & 0.3\\
0.2 & 0.8 & -0.4\\
0.2 & 0.7 & 0.9\end{bmatrix}.
$$

- 每一行表示一个 token；
- 每一列表示一个特征维度；
- $D=3$。

SFDR 在每个特征维度上统计所有 token 的标准差。

论文中的方差损失为：

$$
\mathcal{L}_{\mathrm{var}}=\frac{1}{D}\sum_{d=1}^{D}\max(0,\sigma_0-\sigma_d).
$$

其中：

- $\sigma_d$：第 $d$ 个特征维度的标准差；
- $\sigma_0$：最低标准差阈值。

下面为了演示，取：

$$
\sigma_0=0.5.
$$

### 2. 第一个特征维度

第一列为：

$$
z_{:,1}=(0.2,0.2,0.2,0.2).
$$

均值为：

$$
\mu_1=0.2.
$$

总体方差为：

$$
\sigma_1^2=\frac{1}{4}\sum_{i=1}^{4}(z_{i,1}-\mu_1)^2=0.
$$

因此：

$$
\sigma_1=0.
$$

该特征维度对所有 token 输出完全相同，属于完全低方差维度。

对应的惩罚为：

$$
\ell_1=\max(0,0.5-0)=0.5.
$$

### 3. 第二个特征维度

第二列为：

$$
z_{:,2}=(0.8,0.7,0.8,0.7).
$$

均值为：

$$
\mu_2=\frac{0.8+0.7+0.8+0.7}{4}=0.75.
$$

总体方差为：

$$
\sigma_2^2=\frac{(0.8-0.75)^2+(0.7-0.75)^2+(0.8-0.75)^2+(0.7-0.75)^2}{4}.
$$

因此：

$$
\sigma_2^2=0.0025,\qquad\sigma_2=0.05.
$$

该特征维度有一定变化，但变化幅度较小。

对应的惩罚为：

$$
\ell_2=\max(0,0.5-0.05)=0.45.
$$

### 4. 第三个特征维度

第三列为：

$$
z_{:,3}=(1.1,0.3,-0.4,0.9).
$$

均值为：

$$
\mu_3=\frac{1.1+0.3-0.4+0.9}{4}=0.475.
$$

各元素与均值的偏差为：

$$
(0.625,-0.175,-0.875,0.425).
$$

总体方差为：

$$
\sigma_3^2=\frac{0.625^2+(-0.175)^2+(-0.875)^2+0.425^2}{4}=0.341875.
$$

因此：

$$
\sigma_3=\sqrt{0.341875}\approx0.5847.
$$

由于：

$$
\sigma_3>\sigma_0,
$$

所以第三个特征维度不受惩罚：

$$
\ell_3=\max(0,0.5-0.5847)=0.
$$

### 5. 汇总标准差

三个特征维度的标准差为：

$$
(\sigma_1,\sigma_2,\sigma_3)=(0,0.05,0.5847).
$$

对应的惩罚为：

$$
(\ell_1,\ell_2,\ell_3)=(0.5,0.45,0).
$$

### 6. 计算 $\mathcal{L}_{\mathrm{var}}$

$$
\mathcal{L}_{\mathrm{var}}=\frac{1}{3}(0.5+0.45+0).
$$

因此：

$$
\boxed{\mathcal{L}_{\mathrm{var}}\approx0.3167}.
$$

### 7. 结果解释

| 特征维度 | 特征值 | 标准差 | 惩罚 | 含义 |
|---|---|---:|---:|---|
| 第 1 维 | $(0.2,0.2,0.2,0.2)$ | $0$ | $0.5$ | 完全没有区分能力 |
| 第 2 维 | $(0.8,0.7,0.8,0.7)$ | $0.05$ | $0.45$ | 有变化，但区分能力较弱 |
| 第 3 维 | $(1.1,0.3,-0.4,0.9)$ | $0.5847$ | $0$ | 变化充分，不需要额外惩罚 |

SFDR 的作用不是删除低方差维度，而是通过反向传播促使模型改变 encoder 和 predictor，使低方差维度对不同 token 产生更加不同的响应。

例如，第 1 维可能从：

$$
(0.2,0.2,0.2,0.2)
$$

逐渐变为：

$$
(0.1,0.4,-0.2,0.3).
$$

这样，该维度的标准差会增大，并能携带更多区分不同 token 的信息。

---

## 三、两个损失的区别

### Affinity loss

Affinity loss 关注：

$$
\text{token 与 token 之间的关系}.
$$

它检查：

- 哪些 token 应该相似；
- 哪些 token 应该不相关；
- 学生是否恢复了教师的相对几何结构。

### SFDR

SFDR 关注：

$$
\text{每个特征维度在不同 token 上是否具有足够变化}.
$$

它检查：

- 某个特征维度是否对所有 token 输出相同值；
- 某个通道是否出现低方差；
- 表示空间是否存在特征维度坍缩。

可以简单理解为：

- Affinity loss：约束 token 之间的关系；
- SFDR：约束特征维度本身的多样性。

