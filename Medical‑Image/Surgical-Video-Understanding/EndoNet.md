EndoNet：用于腹腔镜视频识别任务的深度架构
Title: EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos
Authors: Andru P. Twinanda, Sherif Shehata, Didier Mutter, Jacques Marescaux, Michel de Mathelin, Nicolas Padoy
Year: 2016
Type: 原创性算法研究
Link: arXiv
Keywords: 腹腔镜视频、手术流程识别、手术阶段识别、器械出现检测、卷积神经网络、多任务学习、迁移学习、SVM、HHMM、Cholec80
1. 背景与问题手术流程识别是计算机辅助手术中的重要研究方向，可以用于：
自动建立手术视频索引；
监控手术流程；
优化手术室和医护人员调度；
为术中辅助系统提供上下文信息；
识别潜在的手术异常或并发症。
传统方法通常依赖两类信息：
视觉特征：像素、颜色、纹理、形状、梯度、SIFT 和 HOG 等；
器械使用信号：人工标注、RFID 或其他额外设备获得的器械信息。
但是，腹腔镜视频存在摄像机移动、运动模糊、血液遮挡、烟雾以及器械外观变化等问题，传统手工特征难以充分描述这些视觉信息。因此，本文研究的问题是：
能否仅利用腹腔镜视频图像，通过 CNN 自动学习有效的视觉特征，并同时完成手术阶段识别和手术器械出现检测？
2. 论文整体思路作者提出了一个名为 EndoNet 的多任务卷积神经网络。EndoNet 同时完成两个任务：
Tool presence detection：检测图像中是否出现某种手术器械；
Phase recognition：识别当前图像属于哪个手术阶段。
随后，作者使用 SVM 进行阶段分类，再使用 Hierarchical HMM 建模手术阶段之间的时间关系。论文的整体流程如下：
图 1：论文提出的完整处理流程。EndoNet 用于特征提取和器械出现检测，SVM 输出阶段置信度，Hierarchical HMM 进一步利用手术流程的时间约束。
图片来源：原论文 Figure 1。
整体流程可以概括为：72%#alphaxiv-mermaid-_r_3e_-49{font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:16px;fill:#2f2d2c;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#alphaxiv-mermaid-_r_3e_-49 .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#alphaxiv-mermaid-_r_3e_-49 .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#alphaxiv-mermaid-_r_3e_-49 .error-icon{fill:#fbfbfb;}#alphaxiv-mermaid-_r_3e_-49 .error-text{fill:#2f2d2c;stroke:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 .edge-thickness-normal{stroke-width:1px;}#alphaxiv-mermaid-_r_3e_-49 .edge-thickness-thick{stroke-width:3.5px;}#alphaxiv-mermaid-_r_3e_-49 .edge-pattern-solid{stroke-dasharray:0;}#alphaxiv-mermaid-_r_3e_-49 .edge-thickness-invisible{stroke-width:0;fill:none;}#alphaxiv-mermaid-_r_3e_-49 .edge-pattern-dashed{stroke-dasharray:3;}#alphaxiv-mermaid-_r_3e_-49 .edge-pattern-dotted{stroke-dasharray:2;}#alphaxiv-mermaid-_r_3e_-49 .marker{fill:#a42f39;stroke:#a42f39;}#alphaxiv-mermaid-_r_3e_-49 .marker.cross{stroke:#a42f39;}#alphaxiv-mermaid-_r_3e_-49 svg{font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:16px;}#alphaxiv-mermaid-_r_3e_-49 p{margin:0;}#alphaxiv-mermaid-_r_3e_-49 .label{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 .cluster-label text{fill:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 .cluster-label span{color:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 .cluster-label span p{background-color:transparent;}#alphaxiv-mermaid-_r_3e_-49 .label text,#alphaxiv-mermaid-_r_3e_-49 span{fill:#2f2d2c;color:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 .node rect,#alphaxiv-mermaid-_r_3e_-49 .node circle,#alphaxiv-mermaid-_r_3e_-49 .node ellipse,#alphaxiv-mermaid-_r_3e_-49 .node polygon,#alphaxiv-mermaid-_r_3e_-49 .node path{fill:#fdfcfc;stroke:#d7d7d7;stroke-width:1px;}#alphaxiv-mermaid-_r_3e_-49 .rough-node .label text,#alphaxiv-mermaid-_r_3e_-49 .node .label text,#alphaxiv-mermaid-_r_3e_-49 .image-shape .label,#alphaxiv-mermaid-_r_3e_-49 .icon-shape .label{text-anchor:middle;}#alphaxiv-mermaid-_r_3e_-49 .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#alphaxiv-mermaid-_r_3e_-49 .rough-node .label,#alphaxiv-mermaid-_r_3e_-49 .node .label,#alphaxiv-mermaid-_r_3e_-49 .image-shape .label,#alphaxiv-mermaid-_r_3e_-49 .icon-shape .label{text-align:center;}#alphaxiv-mermaid-_r_3e_-49 .node.clickable{cursor:pointer;}#alphaxiv-mermaid-_r_3e_-49 .root .anchor path{fill:#a42f39!important;stroke-width:0;stroke:#a42f39;}#alphaxiv-mermaid-_r_3e_-49 .arrowheadPath{fill:#040404;}#alphaxiv-mermaid-_r_3e_-49 .edgePath .path{stroke:#a42f39;stroke-width:1px;}#alphaxiv-mermaid-_r_3e_-49 .flowchart-link{stroke:#a42f39;fill:none;}#alphaxiv-mermaid-_r_3e_-49 .edgeLabel{background-color:#fbfbfb;text-align:center;}#alphaxiv-mermaid-_r_3e_-49 .edgeLabel p{background-color:#fbfbfb;}#alphaxiv-mermaid-_r_3e_-49 .edgeLabel rect{opacity:0.5;background-color:#fbfbfb;fill:#fbfbfb;}#alphaxiv-mermaid-_r_3e_-49 .labelBkg{background-color:rgba(251, 251, 251, 0.5);}#alphaxiv-mermaid-_r_3e_-49 .cluster rect{fill:#fdfcfc;stroke:#d7d7d7;stroke-width:1px;}#alphaxiv-mermaid-_r_3e_-49 .cluster text{fill:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 .cluster span{color:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:12px;background:#fbfbfb;border:1px solid #d7d7d7;border-radius:2px;pointer-events:none;z-index:100;}#alphaxiv-mermaid-_r_3e_-49 .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#2f2d2c;}#alphaxiv-mermaid-_r_3e_-49 rect.text{fill:none;stroke-width:0;}#alphaxiv-mermaid-_r_3e_-49 .icon-shape,#alphaxiv-mermaid-_r_3e_-49 .image-shape{background-color:#fbfbfb;text-align:center;}#alphaxiv-mermaid-_r_3e_-49 .icon-shape p,#alphaxiv-mermaid-_r_3e_-49 .image-shape p{background-color:#fbfbfb;padding:2px;}#alphaxiv-mermaid-_r_3e_-49 .icon-shape .label rect,#alphaxiv-mermaid-_r_3e_-49 .image-shape .label rect{opacity:0.5;background-color:#fbfbfb;fill:#fbfbfb;}#alphaxiv-mermaid-_r_3e_-49 .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#alphaxiv-mermaid-_r_3e_-49 .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#alphaxiv-mermaid-_r_3e_-49 .node .neo-node{stroke:#d7d7d7;}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].node rect,#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].cluster rect,#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].node polygon{stroke:url(#alphaxiv-mermaid-_r_3e_-49-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].node path{stroke:url(#alphaxiv-mermaid-_r_3e_-49-gradient);stroke-width:1px;}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].node .neo-line path{stroke:#d7d7d7;filter:none;}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].node circle{stroke:url(#alphaxiv-mermaid-_r_3e_-49-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].node circle .state-start{fill:#000000;}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].icon-shape .icon{fill:url(#alphaxiv-mermaid-_r_3e_-49-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3e_-49 [data-look="neo"].icon-shape .icon-neo path{stroke:url(#alphaxiv-mermaid-_r_3e_-49-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3e_-49 :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}腹腔镜视频帧EndoNet器械出现检测视觉特征提取器械置信度特征拼接SVM阶段置信度Hierarchical HMM最终手术阶段序列3. 核心方法3.1 EndoNet 网络结构EndoNet 是 AlexNet 的扩展，包含：
五个卷积层；
两个全连接层 fc6 和 fc7；
器械检测层 fc_tool；
特征融合层 fc8；
阶段识别层 fc_phase。

图 2：EndoNet 网络结构。fc_tool 输出七类器械的出现置信度，并与 fc7 的视觉特征拼接后形成 fc8，用于阶段识别。
图片来源：原论文 Figure 2。
网络结构如下：72%#alphaxiv-mermaid-_r_3f_-25{font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:16px;fill:#2f2d2c;}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#alphaxiv-mermaid-_r_3f_-25 .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#alphaxiv-mermaid-_r_3f_-25 .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#alphaxiv-mermaid-_r_3f_-25 .error-icon{fill:#fbfbfb;}#alphaxiv-mermaid-_r_3f_-25 .error-text{fill:#2f2d2c;stroke:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 .edge-thickness-normal{stroke-width:1px;}#alphaxiv-mermaid-_r_3f_-25 .edge-thickness-thick{stroke-width:3.5px;}#alphaxiv-mermaid-_r_3f_-25 .edge-pattern-solid{stroke-dasharray:0;}#alphaxiv-mermaid-_r_3f_-25 .edge-thickness-invisible{stroke-width:0;fill:none;}#alphaxiv-mermaid-_r_3f_-25 .edge-pattern-dashed{stroke-dasharray:3;}#alphaxiv-mermaid-_r_3f_-25 .edge-pattern-dotted{stroke-dasharray:2;}#alphaxiv-mermaid-_r_3f_-25 .marker{fill:#a42f39;stroke:#a42f39;}#alphaxiv-mermaid-_r_3f_-25 .marker.cross{stroke:#a42f39;}#alphaxiv-mermaid-_r_3f_-25 svg{font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:16px;}#alphaxiv-mermaid-_r_3f_-25 p{margin:0;}#alphaxiv-mermaid-_r_3f_-25 .label{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 .cluster-label text{fill:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 .cluster-label span{color:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 .cluster-label span p{background-color:transparent;}#alphaxiv-mermaid-_r_3f_-25 .label text,#alphaxiv-mermaid-_r_3f_-25 span{fill:#2f2d2c;color:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 .node rect,#alphaxiv-mermaid-_r_3f_-25 .node circle,#alphaxiv-mermaid-_r_3f_-25 .node ellipse,#alphaxiv-mermaid-_r_3f_-25 .node polygon,#alphaxiv-mermaid-_r_3f_-25 .node path{fill:#fdfcfc;stroke:#d7d7d7;stroke-width:1px;}#alphaxiv-mermaid-_r_3f_-25 .rough-node .label text,#alphaxiv-mermaid-_r_3f_-25 .node .label text,#alphaxiv-mermaid-_r_3f_-25 .image-shape .label,#alphaxiv-mermaid-_r_3f_-25 .icon-shape .label{text-anchor:middle;}#alphaxiv-mermaid-_r_3f_-25 .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#alphaxiv-mermaid-_r_3f_-25 .rough-node .label,#alphaxiv-mermaid-_r_3f_-25 .node .label,#alphaxiv-mermaid-_r_3f_-25 .image-shape .label,#alphaxiv-mermaid-_r_3f_-25 .icon-shape .label{text-align:center;}#alphaxiv-mermaid-_r_3f_-25 .node.clickable{cursor:pointer;}#alphaxiv-mermaid-_r_3f_-25 .root .anchor path{fill:#a42f39!important;stroke-width:0;stroke:#a42f39;}#alphaxiv-mermaid-_r_3f_-25 .arrowheadPath{fill:#040404;}#alphaxiv-mermaid-_r_3f_-25 .edgePath .path{stroke:#a42f39;stroke-width:1px;}#alphaxiv-mermaid-_r_3f_-25 .flowchart-link{stroke:#a42f39;fill:none;}#alphaxiv-mermaid-_r_3f_-25 .edgeLabel{background-color:#fbfbfb;text-align:center;}#alphaxiv-mermaid-_r_3f_-25 .edgeLabel p{background-color:#fbfbfb;}#alphaxiv-mermaid-_r_3f_-25 .edgeLabel rect{opacity:0.5;background-color:#fbfbfb;fill:#fbfbfb;}#alphaxiv-mermaid-_r_3f_-25 .labelBkg{background-color:rgba(251, 251, 251, 0.5);}#alphaxiv-mermaid-_r_3f_-25 .cluster rect{fill:#fdfcfc;stroke:#d7d7d7;stroke-width:1px;}#alphaxiv-mermaid-_r_3f_-25 .cluster text{fill:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 .cluster span{color:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:Inter,ui-sans-serif,system-ui,sans-serif;font-size:12px;background:#fbfbfb;border:1px solid #d7d7d7;border-radius:2px;pointer-events:none;z-index:100;}#alphaxiv-mermaid-_r_3f_-25 .flowchartTitleText{text-anchor:middle;font-size:18px;fill:#2f2d2c;}#alphaxiv-mermaid-_r_3f_-25 rect.text{fill:none;stroke-width:0;}#alphaxiv-mermaid-_r_3f_-25 .icon-shape,#alphaxiv-mermaid-_r_3f_-25 .image-shape{background-color:#fbfbfb;text-align:center;}#alphaxiv-mermaid-_r_3f_-25 .icon-shape p,#alphaxiv-mermaid-_r_3f_-25 .image-shape p{background-color:#fbfbfb;padding:2px;}#alphaxiv-mermaid-_r_3f_-25 .icon-shape .label rect,#alphaxiv-mermaid-_r_3f_-25 .image-shape .label rect{opacity:0.5;background-color:#fbfbfb;fill:#fbfbfb;}#alphaxiv-mermaid-_r_3f_-25 .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#alphaxiv-mermaid-_r_3f_-25 .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#alphaxiv-mermaid-_r_3f_-25 .node .neo-node{stroke:#d7d7d7;}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].node rect,#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].cluster rect,#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].node polygon{stroke:url(#alphaxiv-mermaid-_r_3f_-25-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].node path{stroke:url(#alphaxiv-mermaid-_r_3f_-25-gradient);stroke-width:1px;}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].node .neo-line path{stroke:#d7d7d7;filter:none;}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].node circle{stroke:url(#alphaxiv-mermaid-_r_3f_-25-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].node circle .state-start{fill:#000000;}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].icon-shape .icon{fill:url(#alphaxiv-mermaid-_r_3f_-25-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3f_-25 [data-look="neo"].icon-shape .icon-neo path{stroke:url(#alphaxiv-mermaid-_r_3f_-25-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#alphaxiv-mermaid-_r_3f_-25 :root{--mermaid-font-family:"trebuchet ms",verdana,arial,sans-serif;}输入图像AlexNet卷积层fc6fc7fc_tool7类器械置信度fc8特征拼接fc_phase7类手术阶段fc_tool 输出七个器械的出现置信度：
Bipolar；
Clipper；
Grasper；
Hook；
Irrigator；
Scissors；
Specimen bag。
fc8 由两部分组成：
fc7 的 4096 维视觉特征；
fc_tool 的 7 维器械置信度。
因此最终特征维度约为：$$
4096+7=4103.
$$3.2 多任务学习EndoNet 同时优化器械检测任务和手术阶段识别任务。器械检测是七个二分类任务，使用二元交叉熵损失：$$
L_T
=
-\frac{1}{N_iN_t}
\sum_{t=1}^{N_t}
\sum_{i=1}^{N_i}
\left[
k_i^t\log\sigma(v_i^t)
+
(1-k_i^t)\log(1-\sigma(v_i^t))
\right].
$$阶段识别是多分类任务，使用 softmax 交叉熵损失：$$
L_P
=
-\frac{1}{N_i}
\sum_{i=1}^{N_i}
\sum_{p=1}^{N_p}
l_i^p\log\phi(w_i^p).
$$总损失为：$$
L=aL_T+bL_P.
$$论文中设置：$$
a=b=1.
$$多任务学习的核心思想是：
器械出现信息和手术阶段之间存在联系。学习器械检测任务能够帮助网络学习更有辨识力的阶段识别特征。
3.3 迁移学习作者使用在 ImageNet 上预训练的 AlexNet 初始化 EndoNet：textImageNet预训练AlexNet
        ↓
增加器械检测分支和阶段识别分支
        ↓
使用Cholec80进行fine-tuning
        ↓
得到EndoNet
具体步骤为：
保留 AlexNet 的卷积层、fc6 和 fc7；
增加 fc_tool 器械检测分支；
增加 fc_phase 阶段识别分支；
对新增层进行随机初始化；
使用 Cholec80 数据进行 fine-tuning。
这样做的原因是医学视频数据较难获得，人工标注成本较高，直接从随机参数开始训练 CNN 比较困难。3.4 SVM 阶段分类作者将 EndoNet 的 fc8 特征输入 SVM。由于 Cholec80 有七个阶段，因此采用 one-vs-all 方法训练七个二分类 SVM：textSVM 1：是不是 Preparation？
SVM 2：是不是 Calot triangle dissection？
SVM 3：是不是 Clipping and cutting？
SVM 4：是不是 Gallbladder dissection？
SVM 5：是不是 Gallbladder packaging？
SVM 6：是不是 Cleaning and coagulation？
SVM 7：是不是 Gallbladder retraction？
对于一帧图像，七个 SVM 分别输出一个阶段分数：$$
\mathbf{s}=[s_1,s_2,\ldots,s_7].
$$如果只进行单帧分类，可以选择得分最高的阶段：$$
\hat{y}=\arg\max_p s_p.
$$通俗来说，七个 SVM 就像七个评委，分别回答：
“这张图像像不像我负责的那个阶段？”
3.5 HHMM 时间建模SVM 主要分析当前帧，无法充分利用视频中的时间信息，因此作者使用 Hierarchical HMM。在该模型中：模型概念对应内容隐藏状态真实手术阶段观测值SVM 输出的阶段分数状态转移手术阶段之间的转换观测概率某阶段产生某种 SVM 分数的可能性HHMM 的结构如下：
图 3：Cholec80 手术阶段的两层 HHMM 结构。顶层状态表示宏观手术阶段，底层状态表示阶段内部的变化。
图片来源：原论文 Figure 7。
顶层状态表示七个宏观阶段：textP1 Preparation
P2 Calot triangle dissection
P3 Clipping and cutting
P4 Gallbladder dissection
P5 Gallbladder packaging
P6 Cleaning and coagulation
P7 Gallbladder retraction
底层状态表示同一个阶段内部可能存在的不同视觉或操作状态。例如，P3 阶段内部可能经历：text寻找位置 → 放置夹子 → 剪切 → 检查
HHMM 通过以下两类信息判断最终阶段：
当前帧的 SVM 分数；
手术阶段之间的合理转移关系。
因此，如果 SVM 产生：textP1 → P3 → P2 → P3
HHMM 可能根据手术流程将其修正为：textP1 → P1 → P2 → P3
也就是说：
SVM 看“当前这一帧像什么”，HHMM 看“整个手术流程应该是什么”。
4. 实验设置4.1 Cholec80 数据集作者构建了 Cholec80 数据集：
80 个胆囊切除手术视频；
由 13 位外科医生完成；
视频原始帧率为 25 fps；
实验中下采样到 1 fps；
每帧包含手术阶段和器械出现标注；
40 个视频用于 fine-tuning；
40 个视频用于测试。
Cholec80 的数据分布如下：
图 4：Cholec80 数据集中器械出现标注和手术阶段标注的分布。可以观察到不同器械和阶段的样本数量并不均衡。
图片来源：原论文 Figure 4。
Cholec80 包含七个手术阶段：
Preparation；
Calot triangle dissection；
Clipping and cutting；
Gallbladder dissection；
Gallbladder packaging；
Cleaning and coagulation；
Gallbladder retraction。
4.2 EndoVis 数据集作者还使用 EndoVis workflow challenge 数据集进行跨数据集测试：
包含 7 个胆囊切除手术视频；
视频来自另一家医院；
阶段定义与 Cholec80 不完全相同；
主要用于测试模型的泛化能力。
由于 EndoVis 中的器械类别和外观与 Cholec80 不同，作者只进行阶段识别实验，没有进行器械检测实验。4.3 对比方法器械检测任务比较：
DPM；
ToolNet；
EndoNet。
阶段识别任务比较：
Binary tool；
Handcrafted；
Handcrafted + CCA；
AlexNet；
PhaseNet；
EndoNet；
EndoNet-GTbin。
5. 实验结果5.1 器械出现检测在 Cholec80 的 40 个测试视频上：方法平均 APDPM58.4%ToolNet80.9%EndoNet81.0%EndoNet 的器械检测平均 AP 为 81.0%，明显优于 DPM，也略优于只进行器械检测的 ToolNet。这说明：
CNN 特征比传统 DPM 更有效；
多任务学习没有损害器械检测性能；
不需要器械边界框，也可以完成器械出现检测。
5.2 阶段识别结果在不使用 HHMM 的情况下，Cholec80 上的准确率如下：方法准确率Binary tool48.2%Handcrafted44.0%AlexNet59.2%PhaseNet73.0%EndoNet75.2%EndoNet-GTbin75.3%EndoNet 优于：
手工设计的视觉特征；
未 fine-tune 的 AlexNet；
只完成阶段识别的 PhaseNet。
这说明：
CNN 特征优于传统手工特征；
fine-tuning 能够改善手术识别性能；
器械检测辅助任务能够帮助阶段识别。
5.3 HHMM 对阶段识别的影响加入 HHMM 后，阶段识别结果如下：方法离线准确率在线准确率Binary tool69.2%47.5%Handcrafted36.7%32.6%AlexNet76.2%67.2%PhaseNet89.1%78.8%EndoNet92.0%81.7%原论文结果表截图：
表 1：加入 HHMM 后的阶段识别结果。EndoNet 在 Cholec80 上取得 92.0% 的离线准确率和 81.7% 的在线准确率。
图片来源：原论文 Table IV。
EndoNet 的阶段识别准确率由：textHHMM前：75.2%
HHMM后离线：92.0%
HHMM后在线：81.7%
这说明手术流程的时间结构非常重要。离线识别的性能高于在线识别，是因为离线模式可以看到完整视频，并使用未来帧修正当前判断。5.4 训练数据量的影响作者分别使用 10、20、30 和 40 个视频进行 fine-tuning。
图 5：训练视频数量对器械检测和阶段识别性能的影响。随着 fine-tuning 视频数量增加，模型性能总体提高。
图片来源：原论文 Figure 10。
结果表明：
训练视频越多，器械检测性能总体越好；
阶段识别准确率也随训练数据增加而提高；
EndoNet 比单任务网络更能从增加的数据中受益；
医学视频模型对训练数据规模比较敏感。
5.5 跨数据集泛化在 EndoVis 数据集上，加入 HHMM 后的离线准确率为：方法离线准确率Binary tool73.0%Handcrafted46.5%AlexNet79.5%PhaseNet79.7%EndoNet86.0%EndoNet 在 EndoVis 上仍然优于 AlexNet 和 PhaseNet，说明多任务学习获得的视觉特征具有一定泛化能力。不过，EndoVis 的性能低于 Cholec80，说明不同医院、器械、医生和阶段定义会造成明显的数据分布差异。5.6 自动手术视频索引作者使用离线阶段识别结果测试自动视频索引。结果显示：
89% 的阶段边界可以在 30 秒内被检测到；
只有 6% 的阶段边界误差超过 2 分钟。
这说明 EndoNet 可以帮助快速建立手术视频索引，减少人工寻找阶段边界的工作量。5.7 Bipolar 和 clipper 检测作者还测试了 bipolar 和 clipper 两种器械的时间检测效果：
超过 90% 的 bipolar 工具块可以在 5 秒内检测到；
bipolar 的误报率为 3.8%；
80% 的 clipper 工具块可以在 5 秒内检测到；
97% 的 clipper 工具块可以在 30 秒内检测到；
clipper 的误报率为 8.3%。
6. 创新点
使用 CNN 自动学习腹腔镜视频的视觉特征；
提出同时进行器械检测和阶段识别的多任务 CNN；
仅根据腹腔镜视频进行器械出现检测；
使用迁移学习解决医学视频标注数据不足问题；
将器械置信度与视觉特征融合；
使用 SVM 进行阶段分类；
使用 HHMM 建模手术阶段的时间关系；
构建包含 80 个手术视频的 Cholec80 数据集；
对单任务、多任务、时间建模和跨数据集泛化进行了较完整的实验。
7. 局限性
EndoNet 本身不包含时间信息；
CNN、SVM 和 HHMM 是分阶段训练的，不是端到端训练；
数据主要来自同一家医院；
跨医院泛化能力仍然有限；
器械检测只判断器械是否出现，不进行定位；
P5 和 P6 等阶段边界不够清晰；
使用的 AlexNet 结构较旧；
离线识别性能明显高于在线识别，实时应用仍然存在挑战。
8. 关键概念问答记录Q1：什么是迁移学习？迁移学习是指先在大规模数据集上训练模型，再将模型学到的知识迁移到数据量较小的新任务上。在本文中：textImageNet预训练AlexNet
        ↓
学习腹腔镜图像特征
        ↓
使用Cholec80进行fine-tuning
        ↓
完成器械检测和阶段识别
可以把它理解成：一个已经学过高中数学的学生，进入大学后学习物理。他不需要重新学习基本的数字和方程，而是利用已有数学知识解决新的物理问题。Q2：什么是多分类 SVM？论文采用 one-vs-all 方法，将七分类问题拆成七个二分类问题。每一个 SVM 负责回答：text当前图像是不是某一个特定阶段？
七个 SVM 输出七个分数，再根据分数判断当前图像最可能属于哪个阶段。SVM 主要回答：
这一帧图像看起来像哪个阶段？
Q3：什么是 HMM？HMM 是隐马尔可夫模型，用于根据可观察现象推测隐藏状态，同时利用状态之间的时间变化规律。以“根据是否带伞判断天气”为例：
隐藏状态：晴天或雨天；
观测：带伞或不带伞；
状态转移：今天的天气与昨天的天气有关。
在 EndoNet 中：
隐藏状态：真实手术阶段；
观测：SVM 输出的阶段分数；
状态转移：手术阶段之间的流程关系。
Q4：SVM 和 HHMM 有什么区别？模型主要依据作用SVM当前单帧图像判断这一帧像哪个阶段HHMM连续视频帧和流程规律判断整个阶段序列是否合理可以简单记忆为：
SVM 看当前帧，HHMM 看完整流程。
Q5：什么是端到端训练？端到端训练是指从原始输入到最终输出的所有模块一起训练。原论文的流程是：text先训练EndoNet
→ 提取特征
→ 单独训练SVM
→ 单独训练HHMM
因此，最终阶段识别错误不能直接反向传播回 CNN。理想的端到端方法可以是：text连续视频帧
→ CNN或Vision Transformer
→ LSTM或Temporal Transformer
→ 阶段预测
→ 计算最终损失
→ 更新整个网络
9. 个人思考这篇论文最大的启发是：手术视频理解不能只被看作普通图像分类问题。手术阶段和器械之间存在明显关联：text器械信息
    ↓
帮助理解当前操作
    ↓
帮助判断手术阶段
因此，多任务学习可以通过器械检测这个辅助任务，帮助主任务学习更好的视觉表示。此外，EndoNet 加入 HHMM 后，离线准确率从 75.2% 提升到 92.0%，说明时间建模在手术视频理解中非常重要。如果我重新开展这项研究，我会：
使用 ResNet、EfficientNet 或 Vision Transformer 替代 AlexNet；
使用 LSTM、Temporal Convolution 或 Transformer 建模时间信息；
将 CNN、时序模型和阶段分类器进行端到端训练；
进行跨医生、跨医院和跨设备测试；
将器械出现检测扩展为器械定位和姿态估计；
加入手术动作和解剖结构识别任务；
研究模型在阶段边界模糊区域的不确定性。
10. 总结EndoNet 研究的是腹腔镜手术视频中的两个任务：
手术器械出现检测；
手术阶段识别。
作者以 AlexNet 为基础，通过迁移学习和多任务学习构建 EndoNet。网络同时学习器械和阶段相关的视觉特征，并将器械置信度融合到阶段识别特征中。之后，作者使用多分类 SVM 获得每帧的阶段置信度，再使用 HHMM 利用手术流程的时间结构，得到更加稳定的阶段序列。主要结果如下：
Cholec80 上器械检测平均 AP：81.0%；
Cholec80 上阶段识别准确率：

HHMM 前：75.2%；
HHMM 后离线：92.0%；
HHMM 后在线：81.7%；


EndoVis 上阶段识别离线准确率：86.0%；
89% 的阶段边界可以在 30 秒内检测到。
这篇论文的核心思想是：
使用迁移学习从有限医学数据中学习视觉特征，使用多任务学习融合器械和阶段信息，再使用 HHMM 利用手术流程的时间结构，从而提高腹腔镜视频的手术阶段识别效果。