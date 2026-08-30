# The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment: Official Splits and Benchmark

> Title: The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment: Official Splits and Benchmark  
> KEYWORDS：Endoscapes2023；腹腔镜胆囊切除术；手术场景分割；目标检测；临界安全视图；手术数据科学；视频理解  
> 作者：Aditya Murali，Deepak Alapatt，Pietro Mascagni，Armine Vardazaryan，Alain Garcia，Nariaki Okamoto，Guido Costamagna，Didier Mutter，Jacques Marescaux，Bernard Dallemagne，Nicolas Padoy  
> 机构：University of Strasbourg；CNRS；IHU Strasbourg；IRCAD；Fondazione Policlinico Universitario A. Gemelli IRCCS  
> Google Scholar 被引次数：未记录  
> 文献类型：技术报告 / 数据集与基准论文  
> 文献来源：arXiv，版本 3，2024 年 10 月 25 日  
> Venue：Technical Report  
> Year：2024  
> Link：[arXiv](https://arxiv.org/abs/2312.12429)  
> 项目地址：[Endoscapes GitHub](https://github.com/CAMMA-public/Endoscapes)

## 1. 背景与问题

### 1.1 研究背景

腹腔镜胆囊切除术是常见的微创手术，但手术过程中存在胆管损伤等安全风险。临床上通常使用“临界安全视图”（Critical View of Safety，CVS）判断胆囊切除前是否已经充分暴露关键解剖结构。

CVS由三个标准组成：

- C1：能够识别并确认两条结构连接到胆囊；
- C2：肝胆三角已经完成充分解剖；
- C3：胆囊已经从胆囊床上充分分离，从而暴露胆囊板。

只有三个标准同时满足时，当前帧才被认为达到了完整的CVS。

与普通的手术阶段识别和手术器械识别相比，CVS评估需要识别细粒度的解剖结构和复杂的手术语义，因此难度更高。模型不仅要识别图像中的工具，还要判断关键解剖结构是否已经被充分暴露，以及多个结构之间是否满足临床上的安全关系。

### 1.2 现有研究的问题

论文指出，之前的DeepCVS数据集虽然推动了自动化CVS评估研究，但存在两个主要限制：

1. **帧选择存在偏差**  
   研究人员手工挑选“适合评估CVS”的图像，因此训练集和测试集可能不能代表真实手术视频中的连续场景。

2. **数据规模较小**  
   之前的数据集只包含少量手工选择的图像，CVS测试集和分割测试集的规模都比较有限，难以支持稳定、全面的模型比较。

因此，论文的核心问题是：

> 如何构建一个规模更大、按照患者级别划分、同时包含CVS标签、目标检测标签和分割标签的手术视频数据集，并为不同任务提供统一的官方基准？

### 1.3 论文目标

论文提出并系统整理了Endoscapes2023数据集，主要目标包括：

- 提供大量腹腔镜胆囊切除术视频帧；
- 提供三个CVS标准的多标签标注；
- 提供关键解剖结构和手术器械的边界框标注；
- 提供部分图像的实例分割和语义分割标注；
- 建立官方训练集、验证集和测试集划分；
- 对目标检测、实例分割和CVS预测任务给出基准结果；
- 为混合监督、半监督、弱监督和时序建模研究提供统一数据基础。

## 2. 数据来源与数据集结构

### 2.1 数据来源

Endoscapes2023来源于201段腹腔镜胆囊切除术视频。论文并不是从整段手术视频中完全均匀地抽取所有帧，而是首先确定CVS具有临床评估意义的手术区域：

- 起点：三个CVS标准中至少有一个标准被认为可以进行评估；
- 终点：胆囊管或胆囊动脉第一次被夹闭。

在这一区域中，视频按照每秒1帧的频率提取图像，总计得到58,813帧。之后根据不同标注成本，采用不同的标注频率：

- CVS标注：每5秒标注1帧；
- 边界框标注：每30秒标注1帧；
- 分割掩码标注：从部分视频中进一步抽取并标注。

CVS标注由三名临床专家独立完成，最终使用多数投票得到每个标准的共识标签。数据按照视频和患者划分，而不是按照单帧随机划分，以避免同一患者的视频帧同时出现在训练集和测试集中。

### 2.2 数据集组成

Endoscapes2023包括三个主要子数据集：

1. **Endoscapes-CVS201**  
   包含来自201段视频的CVS多标签标注。

2. **Endoscapes-BBox201**  
   包含来自201段视频的关键解剖结构和手术器械边界框标注。

3. **Endoscapes-Seg50**  
   从50段视频中选取部分帧，提供实例分割和语义分割掩码。

### 2.3 原文表1：Endoscapes2023数据集划分

| 子数据集 | 视频数（训练/验证/测试/总计） | 帧数（训练/验证/测试/总计） | 标注类型 | 标注帧数（训练/验证/测试/总计） |
|---|---:|---:|---|---:|
| Endoscapes-CVS201 | 120 / 41 / 40 / 201 | 36,694 / 12,372 / 9,747 / 58,813 | CVS二值多标签 | 6,960 / 2,331 / 1,799 / 11,090 |
| Endoscapes-BBox201 | 120 / 41 / 40 / 201 | 36,694 / 12,372 / 9,747 / 58,813 | 边界框 | 1,212 / 409 / 312 / 1,933 |
| Endoscapes-Seg50 | 30 / 10 / 10 / 50 | 10,380 / 2,310 / 2,250 / 14,940 | 实例分割与语义分割 | 343 / 76 / 74 / 493 |

### 2.4 标注类别

目标检测和分割任务包含6类对象：

- Cystic Plate：胆囊板；
- Hepatocystic Triangle Dissection：肝胆三角解剖区域；
- Cystic Artery：胆囊动脉；
- Cystic Duct：胆囊管；
- Gallbladder：胆囊；
- Tool：手术器械。

CVS任务则包含三个二值标签C1、C2和C3。模型需要分别预测三个标准，也可以进一步根据三个标准是否同时满足判断整体CVS是否达成。

### 2.5 标签分布与标注一致性

在Endoscapes-CVS201测试集中，三个标准在帧级别的达成比例分别为：

- C1：24.0%；
- C2：17.1%；
- C3：27.1%；
- 三个标准同时满足：7.7%。

这说明完整CVS在单帧中出现的比例较低，任务具有明显的类别不平衡问题。

三名专家之间的Cohen's kappa为：

- C1：0.33；
- C2：0.53；
- C3：0.44；
- 整体CVS：0.38。

这些结果说明，即使是临床专家，对于部分边界模糊的手术场景也存在一定分歧。因此，CVS自动评估不仅是视觉识别问题，也包含临床标准解释和标注一致性问题。

## 3. 核心方法与模型结构

### 3.1 论文的模型定位

这篇论文的重点不是提出一个全新的深度学习网络，而是：

1. 构建一个新的手术视频数据集；
2. 设计多种不同标注成本的数据子集；
3. 建立目标检测、实例分割和CVS预测的统一基准；
4. 提供官方数据划分和基线结果。

因此，论文中的“模型结构”主要指用于基准测试的模型和CVS预测流程，而不是一个单独的新模型。

### 3.2 三类研究任务

论文围绕三个任务进行实验：

#### 任务一：目标检测

目标是在视频帧中检测五类解剖结构和一类手术器械。论文比较了：

- Faster R-CNN；
- Cascade R-CNN；
- Deformable DETR。

这些模型都使用COCO预训练权重进行初始化，然后在Endoscapes-BBox201上进行微调。

#### 任务二：实例分割

目标是在目标检测的基础上，对关键解剖结构和手术器械进行像素级分割。论文比较了：

- Mask R-CNN；
- Cascade Mask R-CNN；
- Mask2Former。

模型分别在Endoscapes-Seg50和完整但未公开的Endoscapes-Seg201上训练，用于比较实际低标注场景和更多标注数据下的性能差异。

#### 任务三：CVS预测

CVS预测是一个三个输出的多标签分类问题。模型需要分别预测C1、C2和C3，而不是简单地输出一个“CVS达成/未达成”标签。

论文设置了三种训练条件：

1. 只使用Endoscapes-CVS201的CVS标签；
2. 同时使用Endoscapes-CVS201和Endoscapes-BBox201；
3. 同时使用Endoscapes-CVS201和Endoscapes-Seg50。

这种设计可以研究不同标注成本下的模型效果。

### 3.3 CVS预测方法

论文比较了单帧方法和时空方法。

#### 仅使用CVS标签的方法

- ResNet50；
- ResNet50 with Reconstruction；
- ResNet50-MoCov2。

其中，ResNet50 with Reconstruction在分类任务之外增加了图像重建辅助目标；ResNet50-MoCov2使用MoCov2预训练的ResNet50作为骨干网络。

#### 使用边界框标签的方法

- LayoutCVS；
- DeepCVS；
- ResNet50-DetInit；
- LG-CVS。

LayoutCVS首先使用目标检测器识别对象，再将检测框转换为空间布局并进行CVS预测。DeepCVS将空间布局与原始图像结合。ResNet50-DetInit使用训练好的目标检测器初始化分类模型。LG-CVS进一步将对象的视觉特征、语义属性和对象间关系组织成潜在图结构，用于CVS预测。

#### 时空方法

- DeepCVS-Temporal；
- STRG；
- SV2LSTG。

这些方法利用连续视频帧，而不是只分析单张图像。SV2LSTG首先为每一帧构建单帧图结构，再将多个时间步的图连接起来，最终通过图神经网络进行CVS预测。

### 3.4 评价指标

由于三个CVS标准的类别分布不均衡，论文没有将普通分类准确率作为主要指标，而是使用：

- **CVS mAP**：衡量模型区分正负样本的能力；
- **CVS balanced accuracy**：分别计算每个类别的平衡准确率，再取平均值。

对于目标检测和实例分割，使用：

- Detection mAP@[0.5:0.95]；
- Segmentation mAP@[0.5:0.95]。

### 3.5 原文表2：目标检测基准结果

| 检测器 | Cystic Plate | HC Triangle Dissection | Cystic Artery | Cystic Duct | Gallbladder | Tool | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Faster R-CNN | 8.0 | 19.4 | 12.6 | 15.5 | 62.9 | 61.5 | 30.0 |
| Cascade R-CNN | 7.5 | 18.4 | 12.1 | 13.8 | 61.5 | 62.4 | 29.3 |
| Deformable DETR | 9.3 | 19.9 | 14.9 | 19.0 | 69.6 | 63.7 | 32.7 |

从整体mAP来看，Deformable DETR取得了最高结果，为32.7。胆囊和手术器械的检测明显容易，而胆囊板、肝胆三角、胆囊动脉和胆囊管等细粒度结构的检测更困难。

## 4. 实验与结果

### 4.1 目标检测结果

在Endoscapes-BBox201上，Deformable DETR的总体检测mAP最高，为32.7；Faster R-CNN和Cascade R-CNN分别为30.0和29.3。

各类别之间存在明显差异：

- Gallbladder和Tool的检测效果较好；
- Cystic Plate和Cystic Artery的检测效果较差；
- HC Triangle Dissection和Cystic Duct也比大目标更难识别。

这说明Endoscapes中的主要挑战并不是检测大范围、外观明显的目标，而是定位尺寸较小、边界不清晰、外观变化较大的解剖结构。

### 4.2 实例分割结果

在Endoscapes-Seg50上，Mask2Former的边界框mAP为28.8，分割mAP为26.6，整体表现优于Mask R-CNN和Cascade Mask R-CNN。

当使用标注更多的Endoscapes-Seg201进行训练时，模型性能明显提高：

- Mask R-CNN的分割mAP从22.9提高到28.6；
- Cascade Mask R-CNN的分割mAP从25.6提高到30.0；
- Mask2Former的分割mAP从26.6提高到31.2。

这说明分割任务对标注数量较为敏感，Endoscapes-Seg50可以用于低标注研究，而Endoscapes-Seg201则可以作为较高标注条件下的性能上限参考。

### 4.3 原文表7：CVS预测基准结果

下表列出不同训练标注条件下的CVS mAP和balanced accuracy平均值。

| 训练标签 | 方法 | CVS mAP Avg | CVS bAcc Avg |
|---|---|---:|---:|
| Endoscapes-CVS201 Only | ResNet50 | 51.5 | 63.8 |
| Endoscapes-CVS201 Only | ResNet50 with Reconstruction | 52.1 | 64.1 |
| Endoscapes-CVS201 Only | ResNet50-MoCov2 | 57.4 | 66.7 |
| BBox201 + CVS201 | LayoutCVS | 48.2 | 65.4 |
| BBox201 + CVS201 | DeepCVS | 50.5 | 67.1 |
| BBox201 + CVS201 | ResNet50-DetInit | 60.2 | 70.0 |
| BBox201 + CVS201 | LG-CVS | 63.3 | 74.9 |
| BBox201 + CVS201 | DeepCVS-Temporal | 57.8 | 66.9 |
| BBox201 + CVS201 | STRG | 60.5 | 69.8 |
| BBox201 + CVS201 | SV2LSTG | 65.3 | 70.8 |
| Seg50 + CVS201 | LayoutCVS | 57.1 | 69.4 |
| Seg50 + CVS201 | DeepCVS | 56.0 | 71.2 |
| Seg50 + CVS201 | ResNet50-DetInit | 58.5 | 70.5 |
| Seg50 + CVS201 | LG-CVS | 63.2 | 74.8 |
| Seg50 + CVS201 | DeepCVS-Temporal | 60.5 | 73.5 |
| Seg50 + CVS201 | STRG | 59.0 | 68.4 |
| Seg50 + CVS201 | SV2LSTG | 64.3 | 73.4 |
| Seg201 + CVS201 | LG-CVS | 67.3 | 79.8 |
| Seg201 + CVS201 | SV2LSTG | 69.7 | 82.4 |

### 4.4 CVS预测结果分析

在只使用CVS标签的情况下，ResNet50-MoCov2的平均CVS mAP为57.4，高于普通ResNet50的51.5。这说明更好的自监督预训练初始化能够在标注较少的情况下提升模型表现。

在加入边界框标签后，LG-CVS的平均CVS mAP达到63.3，平均balanced accuracy达到74.9，说明显式建模对象及其关系有助于CVS判断。

在使用Seg50和CVS201时，SV2LSTG的平均CVS mAP达到64.3；在使用更多分割标注的Seg201和CVS201时，SV2LSTG进一步达到69.7的平均CVS mAP和82.4的平均balanced accuracy。

整体来看：

- 仅使用CVS标签也可以完成一定程度的CVS预测；
- 加入目标检测或分割信息后，模型可以利用解剖结构的位置和形状；
- 使用时序信息可以帮助模型理解手术过程的连续变化；
- 使用更多高质量分割标注能够进一步提高性能；
- 结构关系建模和时序建模是CVS预测的重要方向。

### 4.5 低标注实验结果

论文还测试了只使用12.5%和25%训练数据的低标注场景。

- 在12.5%标注量下，ResNet50-MoCov2的CVS mAP为38.9，balanced accuracy为55.5；
- 在25%标注量下，ResNet50-MoCov2的CVS mAP为48.8，balanced accuracy为63.6；
- 使用完整训练集时，ResNet50-MoCov2的CVS mAP为57.4，balanced accuracy为66.7。

这说明增加训练数据通常可以提高性能，但随着数据量增加，性能提升可能逐渐减小。自监督预训练在低标注条件下尤其有帮助。

### 4.6 数据集的主要贡献

Endoscapes2023的贡献主要体现在以下几个方面：

1. **规模更大**  
   数据集包含201段手术视频和58,813个候选帧。

2. **标注类型丰富**  
   同时提供CVS标签、边界框和分割掩码。

3. **模拟真实标注成本**  
   CVS标注较密集，边界框标注较稀疏，分割标注更稀疏，符合实际医学数据标注中不同任务的成本差异。

4. **官方数据划分统一**  
   采用患者级视频划分，避免同一患者的视频帧泄漏到不同数据集。

5. **适合多种研究方向**  
   可以支持监督学习、半监督学习、混合监督学习、低标注学习和时序建模。

## 5. 创新点

### 5.1 数据集层面的创新

论文最大的创新是建立了一个围绕CVS自动评估设计的多层次手术视频数据集。与只提供单一分类标签的数据集相比，Endoscapes同时提供：

- 图像级CVS标签；
- 目标级边界框；
- 像素级分割掩码；
- 视频级和帧级数据划分。

这使得研究人员可以探索从视觉识别到临床安全评估的完整任务链条。

### 5.2 分层标注设计

论文按照标注成本设计了不同层次的标注：

- 大量帧只作为未标注视频帧；
- 较多帧拥有CVS标签；
- 较少帧拥有边界框；
- 更少帧拥有分割掩码。

这种设计更接近医学人工智能项目中的真实情况：临床标签相对容易获得，而精确的像素级分割通常需要更高成本。

### 5.3 统一基准设计

论文不仅发布数据，还给出了三类任务的基线模型和结果，包括：

- 目标检测；
- 实例分割；
- CVS预测。

同时还设置了不同标注条件下的CVS预测实验，使研究人员能够比较不同监督信息对模型的影响。

### 5.4 强调结构关系与时序信息

CVS不是单纯判断某个目标是否存在，而是需要理解多个解剖结构之间的关系以及结构在手术过程中的变化。因此，论文中比较的LG-CVS和SV2LSTG等方法分别引入了：

- 对象之间的语义和视觉关系；
- 多帧之间的时空关系；
- 基于图结构的解剖推理。

实验结果表明，这些信息对于CVS预测具有积极作用。

## 6. 不足与局限性

### 6.1 标注规模仍然不均衡

虽然Endoscapes包含58,813帧，但真正具有CVS标签的只有11,090帧，具有边界框标注的只有1,933帧，具有分割掩码的只有493帧。

因此，数据集总帧数很大，但高质量像素级标注相对有限。对于需要分割掩码的模型来说，训练数据仍然受到限制。

### 6.2 类别分布严重不平衡

完整CVS在测试集中的帧级达成率只有7.7%。部分解剖结构的出现频率也较低，这会导致模型倾向于预测高频类别，并增加少数类目标检测和分割的难度。

### 6.3 专家之间存在标注分歧

整体CVS的Cohen's kappa为0.38，C1、C2和C3的kappa分别为0.33、0.53和0.44。这说明CVS判定本身存在一定主观性，模型性能上限可能受到专家标注一致性的限制。

### 6.4 一些结果并非完全在相同条件下获得

论文中的部分CVS结果来自之前的研究，部分结果则是作者重新运行得到的。特别是使用完整Endoscapes-Seg201的结果更接近性能上限，而公开的Endoscapes-Seg50只覆盖50段视频。

因此，在比较不同方法时，需要注意：

- 使用的标注类型不同；
- 使用的分割数据规模不同；
- 部分结果来自已有论文；
- 不同方法的训练设置和预训练条件可能存在差异。

### 6.5 数据来源的代表性有限

数据集主要来自腹腔镜胆囊切除术，并且与特定医疗机构和手术流程有关。模型能否迁移到不同医院、不同设备、不同术者和不同患者群体，还需要进一步验证。

## 7. 个人思考

### 7.1 这篇论文最值得学习的地方

这篇论文最值得学习的地方不是某一个网络结构，而是数据集设计思路。医学图像数据的核心困难通常不是缺少模型，而是：

- 标注成本高；
- 标注标准复杂；
- 标签质量不稳定；
- 不同研究使用的数据划分不一致；
- 同一数据中不同任务的标注密度不同。

Endoscapes通过分层标注和官方划分，把这些实际问题显式地纳入数据集设计中，使后续研究更加接近真实临床应用条件。

### 7.2 如果自己进行研究，可以如何改进

如果基于Endoscapes继续研究，可以考虑以下方向：

1. **利用未标注帧进行半监督学习**  
   58,813帧中只有一部分具有高成本标注，可以使用一致性正则化、伪标签或教师-学生模型利用未标注数据。

2. **引入更强的时序建模**  
   CVS状态不是孤立存在的，连续视频中结构的显露过程具有明显顺序。可以研究视频Transformer、时序图神经网络或基于状态转移的模型。

3. **建模专家不确定性**  
   当前标签主要通过多数投票得到。未来可以保留三位专家的原始判断，将CVS预测建模为带有标注不确定性的任务，而不是简单的确定性分类。

4. **进行跨中心验证**  
   如果能够引入其他医院和设备的数据，可以测试模型的跨域泛化能力，并研究不同手术风格对模型的影响。

5. **改进小目标和低频结构检测**  
   Cystic Plate、Cystic Artery等类别的检测效果较低，可以使用多尺度特征、专门的数据增强、小目标检测模块或结构先验。

6. **联合学习检测、分割和CVS预测**  
   CVS判断依赖关键解剖结构的识别，因此可以设计多任务模型，让检测、分割和CVS预测相互提供监督。

### 7.3 对实验结果的理解

论文结果说明，CVS预测并不是单纯增加数据量就能完全解决的问题。有效的模型需要同时利用：

- 图像外观；
- 解剖结构位置；
- 不同结构之间的关系；
- 连续视频中的变化；
- 临床专家的标注信息。

从结果上看，单帧模型已经可以取得一定效果，但加入对象关系和时序信息后，性能进一步提高。这说明CVS评估本质上更接近一个“结构化视频理解”任务，而不仅仅是普通图像分类任务。

### 7.4 总体评价

Endoscapes2023是一篇以数据集和基准为核心的技术报告。它没有提出一个完全新的模型，而是通过大规模、多层次标注和统一实验设置，为自动化CVS评估建立了较完整的研究基础。

论文的主要价值可以概括为：

> 将CVS评估从小规模、手工筛选的图像分类问题，推进为包含视频、解剖结构、目标检测、分割和时序推理的综合性手术视觉任务。

## 8. 结论

论文提出了Endoscapes2023数据集，用于研究腹腔镜胆囊切除术中的场景理解和临界安全视图评估。数据集包含201段手术视频、58,813帧候选图像、11,090帧CVS标注、1,933帧边界框标注以及493帧分割标注。

论文通过三个任务建立了基准：

- 目标检测；
- 实例分割；
- CVS预测。

实验表明：

- Deformable DETR在目标检测任务中取得最高整体mAP；
- Mask2Former在公开的Endoscapes-Seg50分割设置中表现最好；
- LG-CVS和SV2LSTG等利用结构关系或时序信息的方法，在CVS预测中优于普通单帧分类模型；
- 增加分割标注和使用完整Seg201数据能够进一步提高CVS预测效果；
- 自监督预训练在低标注条件下具有明显帮助。

Endoscapes2023为后续研究提供了统一的数据划分、标注体系和评价基准，也为半监督学习、混合监督学习、时序建模以及临床安全辅助系统研究提供了数据基础。

## 9. 参考文献

1. Murali A, Alapatt D, Mascagni P, et al. The Endoscapes Dataset for Surgical Scene Segmentation, Object Detection, and Critical View of Safety Assessment: Official Splits and Benchmark. 2024.
2. Alapatt D, Mascagni P, Vardazaryan A, et al. Temporally Constrained Neural Networks: A Framework for Semi-Supervised Video Semantic Segmentation. 2021.
3. Alapatt D, Murali A, Srivastav V, et al. Jumpstarting Surgical Computer Vision. 2023.
4. Mascagni P, Vardazaryan A, Alapatt D, et al. Artificial Intelligence for Surgical Safety: Automatic Assessment of the Critical View of Safety in Laparoscopic Cholecystectomy Using Deep Learning. 2021.
5. Murali A, Alapatt D, Mascagni P, et al. Latent Graph Representations for Critical View of Safety Assessment. 2023.
6. Murali A, Alapatt D, Mascagni P, et al. Encoding Surgical Videos as Latent Spatiotemporal Graphs for Object and Anatomy-Driven Reasoning. 2023.
7. Mascagni P, Alapatt D, Garcia A, et al. Surgical Data Science for Safe Cholecystectomy: A Protocol for Segmentation of Hepatocystic Anatomy and Assessment of the Critical View of Safety. 2021.
8. Chen K, Wang J, Pang J, et al. MMDetection: Open MMLab Detection Toolbox and Benchmark. 2019.
9. Twinanda AP, Shehata S, Mutter D, et al. EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. 2016.
10. Nwoye CI, Padoy N. Data Splits and Metrics for Method Benchmarking on Surgical Action Triplet Datasets. 2022.
