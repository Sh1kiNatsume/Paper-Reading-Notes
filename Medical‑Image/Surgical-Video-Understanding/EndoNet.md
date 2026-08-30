# EndoNet：用于腹腔镜视频识别任务的深度架构
> Title: EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos
> Authors: Andru P. Twinanda, Sherif Shehata, Didier Mutter, Jacques Marescaux, Michel de Mathelin, Nicolas Padoy
> Year: 2016
> Type: 原创性算法研究
> Link: arXiv
> Keywords: 腹腔镜视频、手术流程识别、手术阶段识别、器械出现检测、卷积神经网络、多任务学习、迁移学习、SVM、HHMM、Cholec80

## 背景与问题
手术流程识别是计算机辅助手术中的重要研究方向，可以用于：
- 自动建立手术视频索引；
- 监控手术流程；
- 优化手术室和医护人员调度；
- 为术中辅助系统提供上下文信息；
- 识别潜在的手术异常或并发症。

传统方法通常依赖两类信息：
1. 视觉特征：像素、颜色、纹理、形状、梯度、SIFT 和 HOG 等；
2. 器械使用信号：人工标注、RFID 或其他额外设备获得的器械信息。

但是，腹腔镜视频存在摄像机移动、运动模糊、血液遮挡、烟雾以及器械外观变化等问题，传统手工特征难以充分描述这些视觉信息。

因此，本文研究的问题是：
**能否仅利用腹腔镜视频图像，通过 CNN 自动学习有效的视觉特征，并同时完成手术阶段识别和手术器械出现检测？**

## 论文整体思路
作者提出了一个名为 EndoNet 的多任务卷积神经网络。EndoNet 同时完成两个任务：
1. **Tool presence detection**：检测图像中是否出现某种手术器械；
2. **Phase recognition**：识别当前图像属于哪个手术阶段。

随后，作者使用 SVM 进行阶段分类，再使用 Hierarchical HMM 建模手术阶段之间的时间关系。

整体流程概括：
> 腹腔镜视频帧 → EndoNet（器械出现检测 + 视觉特征提取）→ 器械置信度 + 特征拼接 → SVM输出阶段置信度 → Hierarchical HMM → 最终手术阶段序列

> 图1：论文提出的完整处理流程。EndoNet用于特征提取和器械出现检测，SVM输出阶段置信度，Hierarchical HMM进一步利用手术流程的时间约束。图片来源：原论文 Figure 1。

## 3. 核心方法
### 3.1 EndoNet 网络结构
EndoNet 是 AlexNet 的扩展，包含：
- 五个卷积层；
- 两个全连接层 fc6 和 fc7；
- 器械检测层 fc_tool；
- 特征融合层 fc8；
- 阶段识别层 fc_phase。

> 图2：EndoNet 网络结构。fc_tool 输出七类器械的出现置信度，并与 fc7 的视觉特征拼接后形成 fc8，用于阶段识别。图片来源：原论文 Figure 2。

网络数据流：
> 输入图像 → AlexNet卷积层 → fc6 → fc7
> fc7分支1：fc_tool → 7类器械置信度
> fc7分支2：和fc_tool输出做特征拼接得到fc8 → fc_phase → 7类手术阶段

fc_tool 输出七个器械的出现置信度：
`Bipolar； Clipper； Grasper； Hook； Irrigator； Scissors； Specimen bag`。

fc8 由两部分组成：
- fc7 的 4096 维视觉特征；
- fc_tool 的 7 维器械置信度。

因此最终特征维度约为：
$$4096+7=4103$$

### 3.2 多任务学习
EndoNet同时优化器械检测任务和手术阶段识别任务。

器械检测是七个二分类任务，使用二元交叉熵损失 $L_T$：
$$
L_T = -\frac{1}{N_iN_t} \sum_{t=1}^{N_t} \sum_{i=1}^{N_i} \left[ k_i^t\log\sigma(v_i^t) + (1-k_i^t)\log(1-\sigma(v_i^t)) \right]
$$

阶段识别是多分类任务，使用softmax交叉熵损失 $L_P$：
$$
L_P = -\frac{1}{N_i} \sum_{i=1}^{N_i} \sum_{p=1}^{N_p} l_i^p\log\phi(w_i^p)
$$

总损失为：
$$
L=aL_T+bL_P
$$
论文中设置： $a=b=1$。

多任务学习的核心思想：器械出现信息和手术阶段之间存在联系。学习器械检测任务能够帮助网络学习更有辨识力的阶段识别特征。

### 3.3 迁移学习
作者使用在ImageNet上预训练的AlexNet初始化EndoNet：
> ImageNet预训练AlexNet → 增加器械检测分支和阶段识别分支 → 使用Cholec80进行fine‑tuning → 得到EndoNet

具体步骤：
1. 保留AlexNet的卷积层、fc6和fc7；
2. 增加fc_tool器械检测分支；
3. 增加fc_phase阶段识别分支；
4. 对新增层进行随机初始化；
5. 使用Cholec80数据进行fine‑tuning。

原因：医学视频数据较难获得，人工标注成本较高，直接从随机参数开始训练CNN比较困难。

### 3.4 SVM阶段分类
作者将EndoNet的fc8特征输入SVM。由于Cholec80有七个阶段，采用**one‑vs‑all**方法训练七个二分类SVM：
- SVM1：是不是Preparation？
- SVM2：是不是Calot triangle dissection？
- SVM3：是不是Clipping and cutting？
- SVM4：是不是Gallbladder dissection？
- SVM5：是不是Gallbladder packaging？
- SVM6：是不是Cleaning and coagulation？
- SVM7：是不是Gallbladder retraction？

对于一帧图像，七个SVM分别输出一个阶段分数：
$$\mathbf{s}=[s_1,s_2,\ldots,s_7]$$

如果只进行单帧分类，可以选择得分最高的阶段：
$$\hat{y}=\arg\max_p s_p$$

通俗来说，七个 SVM 就像七个评委，分别回答：“这张图像像不像我负责的那个阶段？”

### 3.5 HHMM 时间建模
SVM 主要分析当前帧，无法充分利用视频中的时间信息，因此作者使用 **Hierarchical HMM（分层隐马尔可夫模型）**。

|模型概念|对应内容|
| ---- | ---- |
|隐藏状态|真实手术阶段|
|观测值|SVM 输出的阶段分数|
|状态转移|手术阶段之间的转换|
|观测概率|某阶段产生某种 SVM 分数的可能性|

> 图3：Cholec80 手术阶段的两层 HHMM 结构。顶层状态表示宏观手术阶段，底层状态表示阶段内部的变化。图片来源：原论文 Figure 7。

顶层状态表示七个宏观阶段：
`P1 Preparation、P2 Calot triangle dissection、P3 Clipping and cutting、P4 Gallbladder dissection、P5 Gallbladder packaging、P6 Cleaning and coagulation、P7 Gallbladder retraction`

底层状态表示同一个阶段内部可能存在的不同视觉或操作状态。
例如P3阶段内部：`寻找位置 → 放置夹子 → 剪切 → 检查`。

HHMM 通过两类信息判断最终阶段：
1. 当前帧的 SVM 分数；
2. 手术阶段之间的合理转移关系。

举例：
如果SVM输出序列：`P1 → P3 → P2 → P3`
HHMM会根据手术流程修正为：`P1 → P1 → P2 → P3`

> SVM看“当前这一帧像什么”，HHMM看“整个手术流程应该是什么”。

## 4. 实验设置
### 4.1 Cholec80 数据集
作者构建了 Cholec80 数据集：
- 80 个胆囊切除手术视频；
- 由 13 位外科医生完成；
- 视频原始帧率为 25 fps；实验中下采样到 1 fps；
- 每帧包含手术阶段和器械出现标注；
- 40 个视频用于 fine‑tuning；40 个视频用于测试。

七个手术阶段：
`Preparation； Calot triangle dissection； Clipping and cutting； Gallbladder dissection； Gallbladder packaging； Cleaning and coagulation； Gallbladder retraction`

> 图4：Cholec80 数据集中器械出现标注和手术阶段标注的分布。不同器械和阶段样本数量不均衡。图片来源：原论文 Figure 4。

### 4.2 EndoVis 数据集
用于跨数据集测试：
- 7 个胆囊切除手术视频；
- 来自另一家医院；
- 阶段定义与 Cholec80 不完全相同；
- 测试模型泛化能力。

器械类别、外观和Cholec80不同，只做阶段识别，不做器械检测。

### 4.3 对比方法
器械检测任务对比：DPM； ToolNet； EndoNet。
阶段识别任务对比：Binary tool； Handcrafted； Handcrafted + CCA； AlexNet； PhaseNet； EndoNet； EndoNet‑GTbin。

## 5. 实验结果
### 5.1 器械出现检测（Cholec80测试集）
|方法|平均AP|
| ---- | ---- |
|DPM|58.4%|
|ToolNet|80.9%|
|EndoNet|81.0%|

结论：
1. CNN特征优于传统DPM；
2. 多任务学习没有损害器械检测性能；
3. 不需要器械边界框就可以完成器械出现检测。

### 5.2 阶段识别结果（不使用HHMM）
|方法|准确率|
| ---- | ---- |
|Binary tool|48.2%|
|Handcrafted|44.0%|
|AlexNet|59.2%|
|PhaseNet|73.0%|
|EndoNet|75.2%|
|EndoNet‑GTbin|75.3%|

说明：CNN特征优于手工特征；finetune提升性能；器械检测辅助任务有利于阶段识别。

### 5.3 HHMM 对阶段识别的影响
|方法|离线准确率|在线准确率|
| ---- | ---- | ---- |
|Binary tool|69.2%|47.5%|
|Handcrafted|36.7%|32.6%|
|AlexNet|76.2%|67.2%|
|PhaseNet|89.1%|78.8%|
|EndoNet|92.0%|81.7%|

EndoNet变化：
- HHMM前：75.2%
- HHMM后离线：92.0%
- HHMM后在线：81.7%

> 离线可以看到完整视频，可用未来帧修正结果，性能高于在线实时识别。证明时序建模对手术视频非常关键。

### 5.4 训练数据量的影响
分别用10、20、30、40个视频做finetune：
- 训练视频越多，器械检测、阶段识别性能越高；
- EndoNet多任务网络相比单任务网络，更能从新增数据获益；
- 医学视频模型对训练数据规模敏感。

> 图5：训练视频数量对器械检测和阶段识别性能的影响。图片来源：原论文 Figure 10。

### 5.5 跨数据集泛化 EndoVis（HHMM，离线准确率）
|方法|离线准确率|
| ---- | ---- |
|Binary tool|73.0%|
|Handcrafted|46.5%|
|AlexNet|79.5%|
|PhaseNet|79.7%|
|EndoNet|86.0%|

多任务特征具备一定泛化；但跨医院、器械、医生带来分布偏移，性能相比Cholec80下降。

### 5.6 自动手术视频索引
- 89% 的阶段边界可以在 30 秒内被检测到；
- 只有 6% 的阶段边界误差超过 2 分钟。
可以显著降低人工查找阶段边界的工作量。

### 5.7 Bipolar 和 clipper 检测
- 超过 90% 的 bipolar 工具块可以在 5 秒内检测到；bipolar 的误报率为 3.8%；
- 80% 的 clipper 工具块可以在 5 秒内检测到；97% 的 clipper 工具块可以在 30 秒内检测到；clipper 的误报率为 8.3%。

## 6. 创新点
1. 使用 CNN 自动学习腹腔镜视频的视觉特征；
2. 提出同时进行器械检测和阶段识别的多任务 CNN；
3. 仅根据腹腔镜视频进行器械出现检测（不需要目标框）；
4. 使用迁移学习解决医学视频标注数据不足问题；
5. 将器械置信度与视觉特征拼接融合；
6. SVM完成阶段分类；HHMM建模手术阶段时序关系；
7. 构建包含80个手术视频的Cholec80数据集；
8. 对单任务、多任务、时序建模、跨数据集泛化做完整实验。

## 7. 局限性
1. EndoNet CNN本身不提取时序信息；
2. CNN、SVM、HHMM分阶段训练，**不是端到端训练**；
3. 数据主要来自同一家医院，跨医院泛化有限；
4. 器械检测仅判断是否出现，不做空间定位；
5. P5、P6阶段边界本身比较模糊；
6. 基础 backbone AlexNet 网络比较老旧；
7. 离线性能远高于在线，真实实时部署仍有挑战。

## 8. 关键概念问答记录
**Q1：什么是迁移学习？**
> 迁移学习：先在大规模公开数据集预训练，再迁移到小样本医学数据集做finetune。
> 本文流程：ImageNet预训练AlexNet → 学习腹腔镜图像特征 → Cholec80 finetune → 器械+阶段识别。

**Q2：什么是多分类SVM？**
> one‑vs‑all，七分类拆成7个二分类SVM；每个SVM判断“当前帧是不是本阶段”；输出7个分数，取最高分作为预测。SVM只看单帧图像。

**Q3：什么是HMM？**
> 隐马尔可夫模型：通过观测信号推测隐藏状态，利用状态之间时序转移约束。
> 本文：隐藏状态=真实手术阶段；观测=SVM输出分数；转移概率=手术流程先后约束。

**Q4：SVM 和 HHMM 的区别？**
|模型|主要依据|作用|
| ---- | ---- | ---- |
|SVM|当前单帧图像|判断这一帧图像像哪个阶段|
|HHMM|连续视频帧+手术流程先验|修正得到合理完整阶段序列|

> SVM看当前帧；HHMM看完整手术流程。

**Q5：什么是端到端训练？**
> 端到端：原始输入到最终输出全部模块联合训练，损失可以反向传播更新全部网络参数。
> EndoNet流程：训练CNN → 提取特征 → 单独训练SVM → 单独训练HHMM；模块分离，不是端到端。
> 理想端到端范式：视频帧 → CNN/Vit → LSTM/Temporal Transformer → 阶段预测 → 损失反向传播全部网络。

## 9. 个人思考
这篇论文最大的启发是：手术视频识别不能简单当成普通图像分类。
器械信息与手术操作、阶段高度耦合；器械检测作为辅助任务，能学到更好的视觉表征。

HHMM将离线准确率从75.2%提升到92.0%，充分证明时序建模在手术视频理解任务中的必要性。

如果复现/继续这项研究，可以改进方向：
1. Backbone替换ResNet/EfficientNet/Vision Transformer，替代老旧AlexNet；
2. 使用LSTM、TCN、Temporal Transformer直接建模时序；
3. CNN+时序模块+分类器联合端到端训练；
4. 加大跨医生、跨医院、跨设备的泛化测试；
5. 器械出现检测进一步升级到器械定位、姿态估计；
6. 增加手术动作、解剖结构识别多任务；
7. 研究阶段模糊边界处模型的不确定性估计。

## 10. 总结
EndoNet针对腹腔镜视频，同时解决**器械出现检测**与**手术阶段识别**。
以AlexNet为骨干，迁移学习+多任务学习，融合器械置信度特征；SVM做单帧阶段打分；HHMM引入手术时序约束输出平滑的阶段序列。

关键指标：
- Cholec80器械检测AP：81.0%
- Cholec80阶段识别：无HHMM 75.2%；HHMM离线92.0%；HHMM在线81.7%
- EndoVis跨数据集离线：86.0%
- 89%阶段边界误差控制在30秒以内。

核心思想：利用迁移学习缓解医学数据稀缺，多任务学习挖掘器械‑阶段之间的关联，HHMM引入手术流程时序先验，提升腹腔镜手术视频阶段识别性能。
