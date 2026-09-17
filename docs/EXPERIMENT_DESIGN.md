# CF-ZeroCRC 实验设计

## 1. 研究问题

目标是在**不使用临床信息、不使用肿瘤 ROI 训练、不训练分割网络**的条件下，利用通用大型文生图扩散模型 Stable Diffusion 3.5 Medium 的条件生成动力学，在结直肠癌 CT 上获得异常区域热图和粗粒度三维候选病灶。

输入为单个患者的 CT NIfTI volume：

\[
V=\{X_1,X_2,\ldots,X_N\}.
\]

输出为连续异常图：

\[
M(x,y,z)\in[0,1],
\]

以及阈值化后的候选异常区域、2D/3D bounding box 和 connected components。

模型输入中**不使用** CEA、CA19-9、年龄、性别、T/N 分期、最大径和病灶位置等临床变量。

---

## 2. 核心假设

如果某个 CT 区域本身已经表现出肿瘤异常，那么在相同 noisy latent 下：

- “恢复正常结直肠结构”的提示词应该要求扩散模型对该区域做更大的生成动力学修正；
- “存在结直肠癌”的提示词应该与原始异常结构更加一致；
- 正常组织则表现出相反或较弱的差异。

因此，不必完全依赖最终生成的 pseudo-healthy CT，可以直接比较 SD3.5 Transformer 在三个文本条件下的空间 flow / denoising prediction。

---

## 3. 基础模型与硬件

主模型：**Stable Diffusion 3.5 Medium**。

- 模型来源：ModelScope `stabilityai/stable-diffusion-3.5-medium`
- 框架：PyTorch + Diffusers
- 推理精度：BF16
- 目标硬件：单张 RTX 4090 48GB
- 第一阶段完全冻结 SD3.5，不使用 LoRA，不做监督训练。

ModelScope 仅用于下载权重；运行时从本地 Diffusers 目录加载，不依赖 Hugging Face 在线访问。

---

## 4. Tri-Prompt Counterfactual Flow

### 4.1 三组文本条件

**Neutral prompt \(P_0\)**：

> An axial abdominal CT image. Preserve the anatomical structures, tissue appearance and CT imaging characteristics.

**Healthy prompt \(P_H\)**：

> An axial abdominal CT image showing normal colorectal bowel wall without focal mass, tumor, irregular wall thickening or malignant lesion. Preserve all other anatomical structures and CT imaging characteristics.

**Abnormal prompt \(P_A\)**：

> An axial abdominal CT image showing colorectal cancer with focal bowel wall thickening or a colorectal mass. Preserve the CT imaging characteristics.

### 4.2 CT latent

经过软组织窗后，将灰度切片复制为三通道，输入 SD3.5 VAE：

\[
z_0=E_{VAE}(X).
\]

采用 deterministic VAE mode，避免 VAE sampling 引入额外随机性。

对所有文本条件固定同一个噪声 \(\epsilon\) 和 timestep：

\[
z_t=\mathrm{ScaleNoise}(z_0,t,\epsilon).
\]

### 4.3 三路 Transformer 响应

\[
v_0=F_\theta(z_t,t,P_0),
\]

\[
v_H=F_\theta(z_t,t,P_H),
\]

\[
v_A=F_\theta(z_t,t,P_A).
\]

三者输入的图像 latent、噪声、timestep 均完全相同，唯一变量为文本条件。

### 4.4 Counterfactual Flow Map

计算：

\[
D_H(x,y)=\|v_H(x,y)-v_0(x,y)\|_2,
\]

\[
D_A(x,y)=\|v_A(x,y)-v_0(x,y)\|_2.
\]

定义核心异常响应：

\[
M_{CF}(x,y)=\operatorname{ReLU}\left(D_H(x,y)-D_A(x,y)\right).
\]

这是本项目与单纯“生成健康 CT 后做像素差”的关键区别。

---

## 5. 多噪声尺度

Fine stage 使用：

\[
s\in\{0.15,0.25,0.35\}.
\]

分别获得：

\[
M_{CF}^{0.15}, M_{CF}^{0.25}, M_{CF}^{0.35}.
\]

经 robust normalization 后求平均：

\[
M_{flow}=\frac{1}{3}\sum_s \operatorname{Norm}(M_{CF}^{s}).
\]

低噪声保留局部细节，中等噪声兼顾结构，高一些的噪声增强文本语义条件影响。

Coarse stage 默认只使用 \(s=0.25\) 降低计算量。

---

## 6. Paired Counterfactual Reconstruction 分支

为了提供与 flow response 独立的证据，Fine stage 对同一 CT 使用相同 seed、相同 strength、相同 sampler，分别得到：

\[
Y_0, Y_H, Y_A.
\]

建议：

- strength = 0.25
- 20 inference steps
- guidance scale = 3.5

像素反事实残差：

\[
R_H=|Y_H-Y_0|,
\]

\[
R_A=|Y_A-Y_0|,
\]

\[
M_{rec}=\operatorname{ReLU}(R_H-R_A).
\]

同时计算 differential local SSIM：

\[
S_H=1-SSIM(Y_H,Y_0),
\]

\[
S_A=1-SSIM(Y_A,Y_0),
\]

\[
M_{SSIM}=\operatorname{ReLU}(S_H-S_A).
\]

这种差分形式用于尽量抵消通用生成模型本身造成的 CT 风格变化。

---

## 7. 多证据融合与随机一致性

单个 seed 的异常图：

\[
M^{(k)}=w_f M_{flow}^{(k)}+w_r M_{rec}^{(k)}+w_s M_{SSIM}^{(k)}.
\]

初始默认：

\[
(w_f,w_r,w_s)=(0.5,0.3,0.2).
\]

这些权重仅作为预注册默认值。正式论文实验应在开发集上冻结，不能根据最终测试集调参。

Fine stage 默认使用三个随机种子：42、123、3407。

\[
\bar M=\frac{1}{K}\sum_k M^{(k)},
\]

\[
C(x,y)=\exp\left(-\frac{Var_k(M^{(k)}(x,y))}{\tau}\right),
\]

\[
M_{fine}=\operatorname{Norm}(\bar M\odot C).
\]

随机种子间不稳定的生成伪影会受到抑制。

---

## 8. CT 预处理

第一版保持简单：

1. NIfTI 转 closest-canonical orientation；
2. 轴位逐层处理；
3. soft-tissue window：默认 \([-160,240]\) HU；
4. 映射到 \([0,1]\)；
5. 单张灰度切片复制为 RGB 三通道；
6. 不采用相邻三层作为 RGB，以避免人为彩色边缘；
7. 使用 HU threshold + 最大连通域获得 label-free body mask。

Body mask 仅用于排除空气背景与扫描床，不提供任何肿瘤先验。

---

## 9. 两阶段推理

### 9.1 Stage A：Coarse Localization

用于 1000 例私有数据的计算控制。

默认：

- 256×256；
- 每隔 2 层扫描；
- 单 seed；
- 单噪声尺度 0.25；
- 只运行 tri-prompt flow，不执行完整重建。

每层定义粗筛 score 为 anomaly map 中 top 5% 像素均值。

选择病例内部高分切片后，向上下各扩展 3 层进入 Fine stage。

该阶段只用于计算加速，不是最终评价协议。

### 9.2 Stage B：Fine Localization

候选切片使用：

- 512×512；
- 3 个 noise levels；
- 3 个 seeds；
- flow branch；
- reconstruction branch；
- differential SSIM；
- seed consistency。

对于 MSD 定量实验，推荐使用 `exhaustive` 模式，对所有有效 body slices 执行 Fine stage，避免 coarse selection 对公平评价造成影响。

---

## 10. Z 轴连续性与 3D 后处理

对逐层异常图沿 z 轴做 1D Gaussian smoothing：

\[
M'(x,y,z)=G_{\sigma_z}*M(x,y,z),
\]

默认 \(\sigma_z=1\)。

阈值化默认使用无需标签的 median + MAD：

\[
T=median(M)+k\cdot1.4826\cdot MAD(M).
\]

随后执行：

- 2D small-component filtering；
- morphological closing；
- 3D connected-component filtering；
- 输出候选区域及 3D bounding boxes。

---

## 11. 数据集与实验协议

### 11.1 公开数据：MSD Task10 Colon

用途：**zero-shot quantitative evaluation**。

公开 segmentation mask 仅用于评价，不用于训练 SD3.5、学习 fusion weights 或优化 prompt。

建议将少量病例设为 development subset，仅用于冻结超参数；最终测试病例不参与任何调参。

### 11.2 私有约 1000 例结直肠癌 CT

用途：外部泛化验证。

模型输入只使用 CT。

理想情况下额外选 50–100 例，由医生提供粗矩形框或少量精细轮廓，仅用于独立评价。

---

## 12. 主要评价指标

### Pixel / voxel level

- AUROC
- AUPRC

### Segmentation-like evaluation

- Dice（作为补充，不建议作为唯一主指标）

### Localization level

- Pointing-game accuracy
- 3D bounding-box IoU
- IoU@0.1 / IoU@0.3 lesion localization success

### Detection level

正式投稿版本建议进一步实现 FROC：

- sensitivity @ 0.5 FP/volume
- sensitivity @ 1 FP/volume
- sensitivity @ 2 FP/volume

---

## 13. Baseline 与消融实验

至少包括：

1. 单 prompt pseudo-healthy reconstruction residual；
2. neutral vs healthy paired residual；
3. 单 prompt flow response；
4. dual-prompt counterfactual flow；
5. **Tri-prompt counterfactual flow**；
6. Tri-prompt + multi-noise；
7. Tri-prompt + reconstruction refinement；
8. Tri-prompt + multi-seed consistency；
9. Tri-prompt + z-axis consistency。

重点消融：

- \(P_H\) vs \(P_0+P_H\) vs \(P_0+P_H+P_A\)；
- flow vs reconstruction vs fusion；
- noise level 0.15 / 0.25 / 0.35 / multi-scale；
- seed 数量 1 / 2 / 3；
- 2D vs 2D + z consistency；
- 256 / 512 输入分辨率。

已有 diffusion zero-shot tumor localization / segmentation 方法（尤其 DiffuGTS）应作为文献对照或可复现实验对照。

---

## 14. Go / No-Go 检验

不要先运行完整 1000 例。

第一阶段建议在 MSD Colon 的 10–20 个带 mask 病例上执行最小实验：

- 只使用 \(P_0,P_H,P_A\)；
- 单 noise level = 0.25；
- 单 seed；
- 只计算 \(M_{CF}=ReLU(D_H-D_A)\)；
- 不做 reconstruction。

如果肿瘤 mask 内的 flow response 相对周围组织没有统计上或视觉上的富集，则说明通用 SD3.5 的 CT domain gap 过大，应暂停完整实验并考虑医学图像 diffusion backbone 或无标签 CT domain adaptation。

如果能够形成稳定病灶热点，再逐步增加 multi-noise、reconstruction、multi-seed 和 z-axis consistency。

---

## 15. 论文方法名称

暂定：**TriCF-CRC**

全称：**Tri-Prompt Counterfactual Diffusion Flow for Training-Free Colorectal Tumor Localization**。

核心贡献应描述为：

> 对相同 noisy CT latent 施加中性、健康和异常三种反事实文本条件，通过比较 SD3.5 MMDiT 的空间生成流响应，而不是仅依赖最终生成图像，实现无需像素级训练的结直肠肿瘤异常区域定位。

---

## 16. 风险与解释边界

1. SD3.5 是通用自然图像生成模型，不保证拥有可靠 CT 医学先验；本研究首先是对这一假设的实证检验。
2. 反事实生成结果不能被解释为真实“健康 CT”。
3. Heatmap 是模型条件响应，不等同于病理学因果证据。
4. 私有病例若全部为癌症，不能据此估计真实临床筛查特异度。
5. 代码和模型输出仅用于研究，不应用于临床诊断。
