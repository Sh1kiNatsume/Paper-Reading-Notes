# Papers
个人论文阅读仓库，包含论文、笔记、摘要与学习总结
# 手术视频领域论文推荐

> 推荐方向：手术视频理解、手术流程识别、手术阶段识别、器械检测、手术动作识别、视觉语言模型、手术场景重建和手术视频生成。

## 手术视频理解 - 早期工作与流程识别

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2008 | [On-line Recognition of Surgical Activity for Monitoring in the Operating Room](https://ieeexplore.ieee.org/document/4798252) | 较早研究在线手术活动识别和术中流程监控 | [Paper](https://ieeexplore.ieee.org/document/4798252)
 | 2010 | [Modeling and Segmentation of Surgical Workflow from Laparoscopic Video](https://link.springer.com/chapter/10.1007/978-3-642-15745-5_49) | 使用视觉特征和器械信息对腹腔镜手术流程进行建模 | [Paper](https://link.springer.com/chapter/10.1007/978-3-642-15745-5_49)
 | 2012 | [Statistical Modeling and Recognition of Surgical Workflow](https://doi.org/10.1016/j.media.2011.10.006) | 使用统计模型和器械信息识别手术流程 | [Paper](https://doi.org/10.1016/j.media.2011.10.006)
 | 2014 | [DeepVideo: Large-scale Video Classification with Convolutional Neural Networks](https://arxiv.org/abs/1412.0767) | 虽然不是专门针对手术，但推动了深度学习视频理解的发展 | [Paper](https://arxiv.org/abs/1412.0767)
 | 2016 | [EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos](https://arxiv.org/abs/1602.03012) | 使用多任务 CNN 同时进行手术阶段识别和器械出现检测 | [Paper](https://arxiv.org/abs/1602.03012)
 | 2018 | [SV-RCNet: Workflow Recognition from Surgical Videos Using Recurrent Convolutional Network](https://ieeexplore.ieee.org/abstract/document/8240734) | 将卷积网络和循环网络结合，用于手术流程识别 | [Paper](https://ieeexplore.ieee.org/abstract/document/8240734)
 | 2019 | [Hard Frame Detection and Online Mapping for Surgical Phase Recognition](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_50) | 关注在线阶段识别中的困难帧检测和时间映射 | [Paper](https://link.springer.com/chapter/10.1007/978-3-030-32254-0_50)
 | 2020 | [Multi-task Recurrent Convolutional Network with Correlation Loss for Surgical Video Analysis](https://arxiv.org/abs/1907.06099) | 通过多任务循环卷积网络和相关性损失进行手术视频分析 | [Paper](https://arxiv.org/abs/1907.06099)

## 手术视频理解 - 阶段识别与时序建模

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2017 | [Deep Learning for Surgical Phase Recognition](https://arxiv.org/abs/1702.03689) | 使用深度学习进行腹腔镜手术阶段识别 | [Paper](https://arxiv.org/abs/1702.03689)
 | 2020 | [TeCNO: Surgical Phase Recognition with Multi-stage Temporal Convolutional Networks](https://arxiv.org/abs/2003.10751) | 使用多阶段时序卷积网络进行在线和离线阶段识别 | [Paper](https://arxiv.org/abs/2003.10751)
 | 2021 | [Temporal Memory Relation Network for Workflow Recognition from Surgical Video](https://arxiv.org/abs/2103.16327) | 使用记忆关系网络增强长时间视频中的流程建模 | [Paper](https://arxiv.org/abs/2103.16327)
 | 2021 | [Trans-SVNet: Accurate Phase Recognition from Surgical Videos via Hybrid Embedding Aggregation Transformer](https://arxiv.org/abs/2103.09712) | 使用 Transformer 聚合局部和全局视频特征 | [Paper](https://arxiv.org/abs/2103.09712)
 | 2021 | [Surgical Workflow Analysis: A Review of the State of the Art](https://doi.org/10.1016/j.media.2021.102245) | 总结手术工作流分析、阶段识别和动作识别方法 | [Paper](https://doi.org/10.1016/j.media.2021.102245)
 | 2022 | [Federated Cycling: Semi-supervised Federated Learning of Surgical Phases](https://arxiv.org/abs/2203.07345) | 使用半监督联邦学习解决多中心手术数据协作问题 | [Paper](https://arxiv.org/abs/2203.07345)
 | 2023 | [SKiT: A Fast Key Information Video Transformer for Online Surgical Phase Recognition](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SKiT_a_Fast_Key_Information_Video_Transformer_for_Online_Surgical_ICCV_2023_paper.html) | 使用关键帧信息和 Transformer 加速在线阶段识别 | [Paper](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SKiT_a_Fast_Key_Information_Video_Transformer_for_Online_Surgical_ICCV_2023_paper.html)
 | 2023 | [Surgical Temporal Action-aware Network with Sequence Regularization for Phase Recognition](https://arxiv.org/abs/2311.12603) | 使用动作感知特征和序列正则化增强阶段识别 | [Paper](https://arxiv.org/abs/2311.12603)
 | 2023 | [MuST: Multi-Scale Transformers for Surgical Phase Recognition](https://arxiv.org/abs/2407.17361) | 使用多尺度 Transformer 建模不同时间范围的手术信息 | [Paper](https://arxiv.org/abs/2407.17361)
 | 2024 | [SurgPLAN: Surgical Phase Localization Network for Phase Recognition](https://arxiv.org/abs/2311.09965) | 将手术阶段识别建模为阶段定位问题 | [Paper](https://arxiv.org/abs/2311.09965)
 | 2024 | [Surgformer: Surgical Transformer with Hierarchical Temporal Attention for Surgical Phase Recognition](https://arxiv.org/abs/2408.03867) | 使用层级时间注意力建模手术视频 | [Paper](https://arxiv.org/abs/2408.03867)
 | 2024 | [Label-guided Teacher for Surgical Phase Recognition via Knowledge Distillation](https://papers.miccai.org/miccai-2024/paper/1997_paper.pdf) | 使用知识蒸馏和标签引导教师模型提升阶段识别 | [Paper](https://papers.miccai.org/miccai-2024/paper/1997_paper.pdf)
 | 2024 | [SR-Mamba: Effective Surgical Phase Recognition with State Space Model](https://arxiv.org/abs/2407.08333) | 使用 Mamba 状态空间模型进行手术阶段识别 | [Paper](https://arxiv.org/abs/2407.08333)
 | 2024 | [SPRMamba: Surgical Phase Recognition for Endoscopic Submucosal Dissection with Mamba](https://arxiv.org/abs/2410.20026) | 将 Mamba 应用于内镜黏膜下剥离术阶段识别 | [Paper](https://arxiv.org/abs/2410.20026)
 | 2024 | [SurgPETL: Parameter-Efficient Image-to-Surgical-Video Transfer Learning for Surgical Phase Recognition](https://arxiv.org/abs/2409.20083) | 研究参数高效的图像到手术视频迁移学习 | [Paper](https://arxiv.org/abs/2409.20083)
 | 2025 | [LoViT: Long Video Transformer for Surgical Phase Recognition](https://www.sciencedirect.com/science/article/pii/S1361841524002913) | 针对长手术视频设计的 Transformer 阶段识别方法 | [Paper](https://www.sciencedirect.com/science/article/pii/S1361841524002913)
 | 2025 | [SWAG: Long-term Surgical Workflow Prediction with Generative-based Anticipation](https://arxiv.org/abs/2412.18849) | 使用生成式方法预测未来手术流程 | [Paper](https://arxiv.org/abs/2412.18849)

## 手术视频理解 - 数据集与基准

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2016 | [Cholec80 / EndoNet Dataset](https://arxiv.org/abs/1602.03012) | 80 个胆囊切除手术视频，包含阶段和器械标注 | [Paper](https://arxiv.org/abs/1602.03012)
 | 2016 | [M2CAI16 Workflow Challenge](https://arxiv.org/abs/1610.09278) | 腹腔镜胆囊切除手术流程识别挑战数据集 | [Paper](https://arxiv.org/abs/1610.09278)
 | 2021 | [CholecT50: A Multi-task Dataset for Surgical Action Triplet Recognition](https://arxiv.org/abs/2109.03223) | 用于器械、动作和目标三元组识别的数据集 | [Paper](https://arxiv.org/abs/2109.03223)
 | 2021 | [HeiChole: A Large-scale Dataset for Surgical Phase Recognition and Skill Assessment](https://arxiv.org/abs/2109.14956) | 同时支持阶段识别、动作识别和手术技能评估 | [Paper](https://arxiv.org/abs/2109.14956)
 | 2022 | [PSI-AVA: A Dataset for Surgical Phase, Step, Action, and Instrument Recognition](https://arxiv.org/abs/2212.04582) | 面向前列腺切除术的阶段、步骤、动作和器械识别数据集 | [Paper](https://arxiv.org/abs/2212.04582)
 | 2023 | [Cholec80-CVS: An Open Dataset for Critical View of Safety Assessment](https://www.nature.com/articles/s41597-023-02073-7) | 面向胆囊切除术安全视野评估的开放数据集 | [Paper](https://www.nature.com/articles/s41597-023-02073-7)
 | 2023 | [Endoscapes: A Surgical Scene Segmentation and Critical View of Safety Dataset](https://arxiv.org/abs/2312.12429) | 用于手术场景分割和 CVS 识别的数据集 | [Paper](https://arxiv.org/abs/2312.12429)
 | 2023 | [MultiBypass140: A Dataset for Laparoscopic Gastric Bypass Surgery](https://arxiv.org/abs/2312.11250) | 面向腹腔镜胃旁路手术的阶段和步骤识别数据集 | [Paper](https://arxiv.org/abs/2312.11250)
 | 2024 | [Pit-Vis: Endoscopic Pituitary Surgery Video Dataset](https://arxiv.org/abs/2409.01184) | 面向内镜垂体手术的步骤和器械识别数据集 | [Paper](https://arxiv.org/abs/2409.01184)
 | 2024 | [EgoSurgery-Phase: A Dataset of Surgical Phase Recognition from Egocentric Open Surgery Videos](https://arxiv.org/abs/2405.19644) | 面向第一视角开放手术视频的阶段识别数据集 | [Paper](https://arxiv.org/abs/2405.19644)
 | 2025 | [SurgBench: A Unified Large-Scale Benchmark for Surgical Video Analysis](https://arxiv.org/abs/2506.07603) | 统一评测手术视频分析任务的大规模基准 | [Paper](https://arxiv.org/abs/2506.07603)
 | 2025 | [SurgVU: Surgical Visual Understanding Dataset](https://arxiv.org/abs/2501.09209) | 面向手术视觉理解的大规模数据集 | [Paper](https://arxiv.org/abs/2501.09209)
 | 2025 | [LEMON: A Large Endoscopic MONocular Dataset and Foundation Model for Perception in Surgical Settings](https://arxiv.org/abs/2503.19740) | 大规模单目内镜数据集和手术感知基础模型 | [Paper](https://arxiv.org/abs/2503.19740)
 | 2025 | [PhaKIR: Video Dataset for Surgical Phase, Keypoint, and Instrument Recognition](https://arxiv.org/abs/2511.06549) | 同时支持阶段、关键点和器械识别 | [Paper](https://arxiv.org/abs/2511.06549)

## 手术视频理解 - 手术动作与 Triplet Recognition
年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2019 | [Surgical Action Recognition with Deep Learning](https://arxiv.org/abs/1905.05172) | 使用深度学习识别手术中的细粒度动作 | [Paper](https://arxiv.org/abs/1905.05172)
 | 2021 | [CholecT50: A Multi-task Dataset for Surgical Action Triplet Recognition](https://arxiv.org/abs/2109.03223) | 将手术活动表示为器械、动作和目标三元组 | [Paper](https://arxiv.org/abs/2109.03223)
 | 2022 | [CholecTriplet2021: A Benchmark Challenge for Surgical Action Triplet Recognition](https://arxiv.org/abs/2204.04746) | 手术动作三元组识别基准和挑战赛 | [Paper](https://arxiv.org/abs/2204.04746)
 | 2024 | [Tail-Enhanced Representation Learning for Surgical Triplet Recognition](https://papers.miccai.org/miccai-2024/paper/0026_paper.pdf) | 针对长尾类别的手术三元组识别方法 | [Paper](https://papers.miccai.org/miccai-2024/paper/0026_paper.pdf)
 | 2025 | [fine-CLIP: Enhancing Zero-Shot Fine-Grained Surgical Action Recognition with Vision-Language Models](https://arxiv.org/abs/2503.19670) | 使用视觉语言模型进行零样本细粒度手术动作识别 | [Paper](https://arxiv.org/abs/2503.19670)
 | 2026 | [Generalized Recognition of Basic Surgical Actions Enables Skill Assessment and Vision-Language-Model-based Surgical Planning](https://arxiv.org/abs/2603.12787) | 基础手术动作识别、技能评估与手术规划 | [Paper](https://arxiv.org/abs/2603.12787)

## 手术器械检测、跟踪与分割

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2014 | [Fast Part-based Classification for Instrument Detection in Minimally Invasive Surgery](https://link.springer.com/chapter/10.1007/978-3-662-44781-5_82) | 使用部件模型进行微创手术器械检测 | [Paper](https://link.springer.com/chapter/10.1007/978-3-662-44781-5_82)
 | 2015 | [Detecting Surgical Tools by Modelling Local Appearance and Global Shape](https://ieeexplore.ieee.org/document/7100311) | 联合局部外观和全局形状检测手术器械 | [Paper](https://ieeexplore.ieee.org/document/7100311)
 | 2017 | [Deep Learning for Surgical Tool Detection and Segmentation](https://arxiv.org/abs/1702.05780) | 使用深度学习进行手术工具检测和分割 | [Paper](https://arxiv.org/abs/1702.05780)
 | 2018 | [Surgical Tool Localization with Deep Learning](https://arxiv.org/abs/1802.04410) | 研究腹腔镜视频中的手术工具定位 | [Paper](https://arxiv.org/abs/1802.04410)
 | 2021 | [Image Compositing for Segmentation of Surgical Tools without Manual Annotations](https://arxiv.org/abs/2102.09528) | 通过图像合成减少手术器械分割的人工标注 | [Paper](https://arxiv.org/abs/2102.09528)
 | 2023 | [SurgToolLoc: Surgical Tool Localization Challenge](https://arxiv.org/abs/2305.07152) | 面向机器人手术器械定位的挑战赛 | [Paper](https://arxiv.org/abs/2305.07152)
 | 2023 | [SurgT: Surgical Tissue Tracking](https://arxiv.org/abs/2302.03022) | 面向手术组织跟踪的数据和任务 | [Paper](https://arxiv.org/abs/2302.03022)
 | 2024 | [EgoSurgery-Tool: A Dataset of Surgical Tool and Hand Detection from Egocentric Open Surgery Videos](https://arxiv.org/abs/2406.03095) | 第一视角开放手术中的器械和手部检测 | [Paper](https://arxiv.org/abs/2406.03095)
 | 2024 | [CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools](https://arxiv.org/abs/2312.07352) | 多视角手术器械跟踪数据集 | [Paper](https://arxiv.org/abs/2312.07352)
 | 2024 | [SURGIVID: Annotation-Efficient Surgical Video Object Discovery](https://arxiv.org/abs/2409.07801) | 研究标注高效的手术视频对象发现 | [Paper](https://arxiv.org/abs/2409.07801)

## 手术场景理解、分割与安全评估

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2022 | [Towards Holistic Surgical Scene Understanding](https://arxiv.org/abs/2212.04582) | 从多个层次理解器械、组织、动作和手术步骤 | [Paper](https://arxiv.org/abs/2212.04582)
 | 2023 | [Endoscapes: A Surgical Scene Segmentation and Critical View of Safety Dataset](https://arxiv.org/abs/2312.12429) | 手术场景分割、器械识别和安全视野判断 | [Paper](https://arxiv.org/abs/2312.12429)
 | 2024 | [OSSAR: Towards Open-Set Surgical Activity Recognition in Robot-assisted Surgery](https://arxiv.org/abs/2402.06985) | 研究机器人辅助手术中的开放集活动识别 | [Paper](https://arxiv.org/abs/2402.06985)
 | 2024 | [SANGRIA: Surgical Video Scene Graph Optimization for Surgical Workflow Prediction](https://arxiv.org/abs/2407.20214) | 使用手术场景图预测未来工作流 | [Paper](https://arxiv.org/abs/2407.20214)
 | 2025 | [Surgical Scene Understanding in the Era of Foundation AI Models](https://arxiv.org/abs/2502.14886) | 总结基础模型时代的手术场景理解方法 | [Paper](https://arxiv.org/abs/2502.14886)
 | 2025 | [Surgical Visual Understanding Dataset](https://arxiv.org/abs/2501.09209) | 面向手术视觉理解的多任务数据集 | [Paper](https://arxiv.org/abs/2501.09209)

## 手术视觉问答与视觉语言模型

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2022 | [Surgical VQA: Visual Question Answering for Surgical Videos](https://arxiv.org/abs/2206.11053) | 将视觉问答引入手术视频理解 | [Paper](https://arxiv.org/abs/2206.11053)
 | 2023 | [SSG-VQA: Surgical Scene Graph-based Visual Question Answering](https://arxiv.org/abs/2312.10251) | 基于手术场景图进行视觉问答 | [Paper](https://arxiv.org/abs/2312.10251)
 | 2023 | [EndoChat: A Large-scale Multimodal Dataset for Endoscopic Vision-Language Learning](https://arxiv.org/abs/2501.11347) | 面向内镜图像和文本理解的多模态数据集 | [Paper](https://arxiv.org/abs/2501.11347)
 | 2024 | [PSI-AVA-VQA: Visual Question Answering in Prostatectomy Videos](https://arxiv.org/abs/2304.09974) | 面向前列腺切除术的视觉问答任务 | [Paper](https://arxiv.org/abs/2304.09974)
 | 2025 | [SurgVLM: A Vision-Language Model for Surgical Video Understanding](https://arxiv.org/abs/2506.02555) | 面向多种手术任务的视觉语言模型和评测基准 | [Paper](https://arxiv.org/abs/2506.02555)
 | 2025 | [SurgPub-Video: A Comprehensive Surgical Video Dataset for Vision-Language Models](https://arxiv.org/abs/2508.10054) | 面向视觉语言模型的综合手术视频数据集 | [Paper](https://arxiv.org/abs/2508.10054)
 | 2026 | [CliPPER: Contextual Video-Language Pretraining on Long-form Intraoperative Surgical Procedures](https://arxiv.org/abs/2603.24539) | 面向长时手术视频事件识别的视频语言预训练 | [Paper](https://arxiv.org/abs/2603.24539)

## 手术场景重建与 3D 理解

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2022 | [Neural Rendering for Stereo 3D Reconstruction of Deformable Tissues in Robotic Surgery](https://arxiv.org/abs/2206.15255) | 使用神经渲染进行机器人手术中可变形组织的三维重建 | [Paper](https://arxiv.org/abs/2206.15255)
 | 2023 | [EndoNeRF: Neural Radiance Fields for 3D Reconstruction of Deformable Tissues](https://arxiv.org/abs/2206.15255) | 将 NeRF 应用于内窥镜手术场景重建 | [Paper](https://arxiv.org/abs/2206.15255)
 | 2023 | [EndoSurf: Neural Surface Reconstruction of Deformable Tissues with Stereo Endoscope Videos](https://arxiv.org/abs/2307.11307) | 使用双目内窥镜视频进行可变形组织表面重建 | [Paper](https://arxiv.org/abs/2307.11307)
 | 2024 | [Deform3DGS: Flexible Deformation for Fast Surgical Scene Reconstruction with Gaussian Splatting](https://arxiv.org/abs/2405.17835) | 使用 3D Gaussian Splatting 快速重建可变形手术场景 | [Paper](https://arxiv.org/abs/2405.17835)
 | 2024 | [Free-SurGS: SfM-Free 3D Gaussian Splatting for Surgical Scene Reconstruction](https://papers.miccai.org/miccai-2024/paper/1818_paper.pdf) | 不依赖传统 SfM 的手术场景 Gaussian Splatting 重建 | [Paper](https://papers.miccai.org/miccai-2024/paper/1818_paper.pdf)
 | 2024 | [LGS: A Light-weight 4D Gaussian Splatting for Efficient Surgical Scene Reconstruction](https://arxiv.org/abs/2406.16073) | 使用轻量化 4D Gaussian Splatting 建模动态手术场景 | [Paper](https://arxiv.org/abs/2406.16073)
 | 2024 | [SurgicalGaussian: Deformable 3D Gaussians for High-Fidelity Surgical Scene Reconstruction](https://papers.miccai.org/miccai-2024/paper/1818_paper.pdf) | 使用可变形 3D Gaussian 表示高保真手术场景 | [Paper](https://papers.miccai.org/miccai-2024/paper/1818_paper.pdf)

## 手术视频生成与世界模型

年份 | 名字 | 简介 | 原文链接
---:|---|---|---
 | 2024 | [Surgical Video Generation: From Diffusion to World Models](https://arxiv.org/abs/2608.26214) | 总结扩散模型和世界模型在手术视频生成中的应用 | [Paper](https://arxiv.org/abs/2608.26214)
 | 2025 | [SAW: Toward a Surgical Action World Model via Controllable and Scalable Video Generation](https://arxiv.org/abs/2603.13024) | 通过可控视频生成构建手术动作世界模型 | [Paper](https://arxiv.org/abs/2603.13024)
 | 2025 | [Cosmos-H-Surgical: Learning Surgical Robot Policies from Videos via World Modeling](https://arxiv.org/abs/2512.23162) | 使用世界模型从手术视频学习机器人策略 | [Paper](https://arxiv.org/abs/2512.23162)
 | 2026 | [SurgMotion: A Video-Native Foundation Model for Universal Understanding of Surgical Videos](https://arxiv.org/abs/2602.05638) | 面向通用手术视频理解的视频原生基础模型 | [Paper](https://arxiv.org/abs/2602.05638)
 | 2026 | [Scaling Video Pretraining for Surgical Foundation Models](https://arxiv.org/abs/2603.29966) | 研究扩大手术视频预训练规模的方法 | [Paper](https://arxiv.org/abs/2603.29966)

## 推荐阅读顺序

主要背景阅读论文记录：

1. [EndoNet](https://arxiv.org/abs/1602.03012)：了解手术阶段识别的基本问题；
2. [SV-RCNet](https://ieeexplore.ieee.org/abstract/document/8240734)：了解 CNN 和 RNN 如何结合；
3. [TeCNO](https://arxiv.org/abs/2003.10751)：了解时序卷积网络；
4. [TMRNet](https://arxiv.org/abs/2103.16327)：了解长时间依赖建模；
5. [Trans-SVNet](https://arxiv.org/abs/2103.09712)：了解 Transformer 在手术视频中的应用；
6. [CholecT50](https://arxiv.org/abs/2109.03223)：了解手术动作三元组；
7. [Endoscapes](https://arxiv.org/abs/2312.12429)：了解手术场景理解和安全视野评估；
8. [SKiT](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SKiT_a_Fast_Key_Information_Video_Transformer_for_Online_Surgical_ICCV_2023_paper.html)：了解在线阶段识别；
9. [Surgformer](https://arxiv.org/abs/2408.03867)：了解层级时间注意力；
10. [SurgBench](https://arxiv.org/abs/2506.07603)：了解大规模手术视频基准；
11. [SurgVLM](https://arxiv.org/abs/2506.02555)：了解视觉语言模型在手术视频中的应用；
12. [SurgMotion](https://arxiv.org/abs/2602.05638)：了解最新的手术视频基础模型方向。

## 方向之间的关系

```text
视频帧
  ↓
视觉特征提取
  ↓
器械检测 / 组织分割 / 解剖结构识别
  ↓
手术动作识别 / 阶段识别
  ↓
手术流程预测 / 技能评估
  ↓
视觉语言问答 / 手术规划
  ↓
机器人控制 / 手术世界模型
```
