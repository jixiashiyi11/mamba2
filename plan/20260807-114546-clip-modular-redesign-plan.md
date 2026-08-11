# CLIP Supervised Mask Transfer 最终执行计划

创建时间：2026-08-07 11:45:46  
更新时间：2026-08-10

## 0. 最终任务定义

本项目当前主线已从 normal-only 改为 supervised source-domain mask training：

```text
Supervised Source-domain Mask Transfer for CLIP-based Anomaly Detection and Localization
```

新的设定：

```text
使用非目标域 source dataset 的异常图像和像素级 mask 训练；
不使用目标域训练图像；
不使用目标域异常 mask；
不使用 synthetic anomaly；
backbone 从 BiomedCLIP 改为标准 CLIP；
保留原 MAMBAZero / ARCC 思路中的 text、CNN、Mamba、ARCC 模块。
```

目标：

```text
学习 source-domain mask-supervised localization 能力，
并验证它能否迁移到未参与训练的目标域。
```

模型最终输出：

```text
1. image-level anomaly score
2. pixel-level anomaly heatmap
```

论文里建议强调：

```text
supervised source-domain mask transfer
CLIP patch-text alignment
feature-level CNN refinement
Mamba response-level global context
ARCC response-level calibration
```

旧的 normal-only 实验保留为历史 ablation，不再作为主线。

## 0.1 与原 BiomedCLIP / MAMBAZero 的关系

保持一致的部分：

```text
Text branch：
normal / abnormal text prototypes 提供语义方向。

CNN branch：
负责局部纹理、边界、形态和空间连续性。

CNN local branch：
从 CLIP 中低层 patch tokens 提取局部纹理、边界和形态修正量。
它只做 feature-level residual refinement，不直接输出 heatmap。

Mamba response context：
位于 A_raw 之后，不接收 text prompt，
输入 CLIP layer 24 last_patch token。
它不接收 CNN residual 之后的 refined_patch，
也不接收 text prototype。
建模全局响应上下文 G_mamba。

ARCC：
接收 A_raw、Mamba global context 和 refined patch feature，
负责最终 response-aware local calibration，
是当前主线唯一直接输出 A_final 的模块。

Loss：
沿用 mask-supervised localization 的 BCE + Dice，
保留 normal top-k suppression、token consistency、ARCC calibration regularization。
针对 screw 前景整体发红问题，新增 mask 外 top-k suppression，
默认 supervised_outside_topk_weight = 0.5。
```

不一样的部分：

```text
Backbone：
BiomedCLIP -> standard CLIP / AA-CLIP ViT-L-14-336。

Training data：
不再 normal-only；
使用非目标域真实异常 mask 监督。

Synthetic anomaly：
不再使用 synthetic image / synthetic mask / morphology generator。

CNN output：
原 BiomedCLIP 中 CNN decoder 可直接产生 cnn_logits；
当前 CLIP 版本不让 CNN 直接输出 heatmap，
而是用 CNN 产生 local_delta 修正 CLIP 最后一层 patch token，
再用 refined patch 和文本原型相似度得到 A_raw。
```

## 1. 数据集最终方案

主实验使用工业异常检测数据，backbone 使用标准 CLIP。

| 阶段 | 数据集 | 使用内容 | 异常图 | GT Mask |
| --- | --- | --- | --- | --- |
| Source Train | 非目标域 source dataset | normal + anomaly | 使用 | 使用 |
| Target Test 1 | VisA / BTAD / MPDD / MVTec-AD 等目标域 | normal + anomaly | 仅测试 | 仅评价 |

主协议：

```text
source dataset with masks -> unseen target dataset
```

严格禁止：

```text
目标数据集图像参与训练
目标数据集异常 mask 参与训练
目标数据集用于挑选最优 checkpoint
synthetic anomaly 参与训练
```

允许：

```text
source dataset 的真实 abnormal images
source dataset 的真实 pixel-level masks
source dataset 的 normal images
```

当前最省事实现：

```text
运行 tools/build_supervised_meta.py，
从 data/mvtec/meta.json 生成 data/mvtec/meta_supervised.json。
新 train split = 原 train normal + 原 test abnormal/mask。
配置 configs/clip_ad/clip_ad_supervised_mask_full.py 默认读取 meta_supervised.json。
```

## 2. 整体方法主线

当前 supervised 核心逻辑：

```text
Frozen CLIP image/text encoder
  -> multi-layer patch tokens: layer 12 / 18 / 24
  -> layer 12 / 18 patch tokens -> CNN local branch -> local_delta
  -> layer 24 patch token + alpha * local_delta
  -> refined layer-24 patch tokens
  -> refined patch 和 normal/abnormal text prototypes 只计算一次 A_raw
  -> Mamba response context, input = layer 24 last_patch token
  -> G_mamba global response context
  -> ARCCCalibration, input = refined patch feature + G_mamba + A_raw
  -> A_final
  -> source-domain mask-supervised loss
```

硬约束：

```text
CNN 不直接输出最终 heatmap。
Mamba/CSSD 不和 CNN 分别输出 heatmap 再相加。
text prompt 不进入 CNN。
text prompt 不进入 feature adapter。
text prompt 只用于 refined patch 之后的 similarity scoring。
Mamba 只提供 global response context，不直接输出最终 heatmap。
Mamba 不接收 text prompt，也不接收 CNN residual 之后的 refined_patch。
Mamba 输入 CLIP layer 24 last_patch。
ARCC 是唯一直接输出 A_final 的 map-level calibration。
```

训练 debug 指标：

```text
dbg_refine_cos:
refined_patch 与 layer24_patch 的平均 cosine。
太接近 1 说明 CNN residual 几乎没改动；太低说明 adapter 可能破坏 CLIP text alignment。

dbg_refine_delta_l2:
最终 residual 修改幅度。
如果长期接近 0，可以增大 adapter_scale 或检查 CNN 是否训练到。

dbg_mamba_context_cos / dbg_mamba_context_delta_l2:
ARCC 实际使用的 fused context 与 refined_patch 的相似度和差异。
这里不表示 Mamba 修改了 patch feature，
而是用来观察 G_mamba 注入 ARCC 后是否过强或过弱。

dbg_local_delta_l2 / dbg_local_delta_abs:
CNN 从 layer 12/18 产生的局部修正强度。
如果很小，说明中低层局部信息没有被用起来。

dbg_l12_last_cos / dbg_l18_last_cos:
layer 12/18 与 layer 24 patch token 的相似度。
如果 layer 12 太低，可能语义差距太大；可试只用 layer 18。

dbg_a_raw_mean / dbg_a_raw_max:
ARCC 前的 text similarity anomaly response。
用来判断 refined_patch 和 text prototype 是否已经能产生可训练响应。

dbg_a_final_mean / dbg_a_final_max:
ARCC 后最终 anomaly response。
和 A_raw 对比看 ARCC 是否过度放大或压低。

dbg_arcc_delta_abs / dbg_arcc_delta_ratio:
ARCC 修改 A_raw 的幅度。
ratio 太大说明 ARCC 可能主导结果；太小说明 ARCC 几乎没起作用。

dbg_g_cal_abs / dbg_arcc_lambda:
ARCC calibration map 强度和可学习系数。

dbg_s_global / dbg_topk_score:
image_score 两个来源，用来判断 image-level 分数主要由 global CLIP 还是 pixel top-k 主导。
当前主配置默认 image_score_topk_ratio = None，
即 image_score 使用 A_final 的 max aggregation。

dbg_topk_score_max / dbg_topk_score_top1 / dbg_topk_score_top5:
同一张 A_final 在三种图像级聚合下的局部异常证据。
max 是单点最大响应，top1 是 top 1% pixel mean，top5 是 top 5% pixel mean。

dbg_image_score_max / dbg_image_score_top1 / dbg_image_score_top5:
S_global + beta * 对应局部聚合分数。
用于定位 pixel metric 高但 image metric 低时，
问题来自 global CLIP 分数、局部聚合策略，还是 A_final 本身。

dbg_mamba_prior_mean / dbg_mamba_prior_max:
Mamba last-patch global context branch 的全局先验强度。
```

对应代码：

```text
configs/clip_ad/clip_ad_supervised_mask_full.py
model/clip_ad.py
model/modules/adapters.py
model/mambaad.py
trainer/clip_ad_trainer.py
data/folder_ad.py
tools/visualize_clip_ad_heatmaps.py
```

可视化诊断：

```text
测试结束会保存 A_raw、A_final、G_cal、G_mamba prior、image score variants 到 npz。
运行 tools/visualize_clip_ad_heatmaps.py 可以生成若干 PNG，
同时显示 image / mask / A_raw / A_final / overlay / G_cal / G_mamba / histogram / red-area ratio。
重点检查 red_ratio_06 / red_ratio_08：
如果正常图和异常图都很高，说明 heatmap 可能全图偏红，image-level score 会不稳定；
如果 pixel AUROC 高但 image AUROC 低，再比较 max / top1 / top5 image score variant。
```

训练监督：

```text
L_mask_bce = BCEWithLogits(A_final, source_mask)
L_mask_dice = Dice(sigmoid(A_final), source_mask)
L_raw_bce = BCEWithLogits(A_raw, source_mask)
L_image = BCEWithLogits(image_score, image_label)
L_normal_topk = normal image 上 top-k patch response suppression
L_outside_topk = mask 外最高响应区域 suppression
L_consistency = 1 - cosine(refined_patch, raw_patch)
L_arcc_cal = ||G_cal||^2
```

总 loss：

```text
L = L_mask_bce
  + L_mask_dice
  + 0.2 * L_raw_bce
  + 0.1 * L_image
  + 0.1 * L_normal_topk
  + 0.5 * L_outside_topk
  + 0.1 * L_consistency
  + 0.01 * L_arcc_cal
```

当前 supervised mask full sanity check 结果：

```text
Config:
configs/clip_ad/clip_ad_supervised_mask_full.py

Run:
runs/CLIPADTrainer_configs_clip_ad_clip_ad_supervised_mask_full_20260809-211040

Avg:
Image AUROC: 98.361
Image AP:    99.362
Image F1:    97.677
AUPRO:       85.813
Pixel AUROC: 97.205
Pixel AP:    80.572
Pixel F1:    78.116
```

解释：

```text
这说明 supervised mask loss 已经成功驱动 pixel-level localization。
相比旧 normal-only 路线，pixel 指标大幅提升。

但如果 meta_supervised.json 是由同一个 MVTec 的 original test abnormal/mask
加入 train，同时仍在 original test 上评价，
这只是 pipeline sanity check，不能作为最终论文主结果，
因为训练和测试异常图像发生重叠。

正式主实验必须改成：
source dataset/category with masks -> unseen target dataset/category test。
```

### 2.1 下一步正式实验：先跨类别，再跨数据集

先做 MVTec 内部跨类别，是因为它最省事：

```text
Source train:
其它 MVTec 类别的 train normal
+ 其它 MVTec 类别的 test abnormal/mask

Target test:
留出的目标类别 original test normal + abnormal/mask

禁止：
目标类别的 train normal 进入训练
目标类别的 abnormal/mask 进入训练
```

对应工具：

```text
tools/build_cross_category_meta.py
```

对应配置：

```text
configs/clip_ad/clip_ad_supervised_mask_cross_category.py
```

第一轮建议先留出 `screw`：

```bash
python tools/build_cross_category_meta.py \
  --root data/mvtec \
  --input meta.json \
  --target-classes screw \
  --output meta_cross_category_screw.json

CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/clip_ad/clip_ad_supervised_mask_cross_category.py \
  -m train \
  --seed 42
```

如果换目标类别，只需要重新生成 meta，并通过环境变量指定：

```bash
python tools/build_cross_category_meta.py \
  --root data/mvtec \
  --input meta.json \
  --target-classes capsule \
  --output meta_cross_category_capsule.json

CLIP_AD_META=meta_cross_category_capsule.json CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/clip_ad/clip_ad_supervised_mask_cross_category.py \
  -m train \
  --seed 42
```

跨类别跑通后，再做跨数据集：

```text
MVTec source masks -> VisA / BTAD / MPDD target test
或 VisA source masks -> MVTec target test
```

旧 normal-only 核心逻辑如下，后续只作为历史记录 / 消融参考：

```text
Text Semantic Evidence + Normality Deviation
  -> Raw Anomaly Map
  -> ARCC Calibration
  -> Final Anomaly Map
```

完整流程：

```text
                    Frozen CLIP
                         |
           +-------------+-------------+
           |                           |
           v                           v
   Global Image Feature        Multi-level Patch Features
           |                           |
           |                  +--------+---------+
           |                  |                  |
           |                  v                  v
           |          Text Similarity      Normality Deviation
           |                  |                  |
           |                  +--------+---------+
           |                           |
           |                           v
           |                    Raw Anomaly Map
           |                           |
           |                          ARCC
           |                           |
           |                    Final Anomaly Map
           |                           |
           v                           v
   Global Anomaly Score        Top-K Patch Score
           +-------------+-------------+
                         |
                         v
                  Image Anomaly Score
```

一句话版本：

```text
只用源域正常图学习 normality，用 CLIP 文本提供异常语义，
在未见目标数据上产生初始异常图，
再用 ARCC 根据局部与全局上下文校准异常响应，
最终同时输出图像级异常分数和像素级异常热力图。
```

## 3. CLIP Backbone 使用方式

CLIP backbone 尽量保持冻结。

输入图像：

```text
x -> CLIP Image Encoder
```

提取两类特征：

```text
V_g：global image embedding
F_1, F_2, ..., F_L：多层 patch features
```

不要只使用最后一层 patch feature。

原因：

```text
浅层：保留 texture / edge / local detail
深层：保留 semantic information
```

最后把多个尺度的 patch features 统一投影到 CLIP embedding space，用于异常定位。

## 4. Text Branch：负责“它像不像异常”

Text branch 使用 CLIP 原始 Text Encoder。

第一版先不要学习 abnormal prompt，使用固定 prompt 模板。

例如类别 `bottle`：

Normal prompts：

```text
a photo of a normal bottle
a photo of an intact bottle
a photo of a flawless bottle
a photo of an undamaged bottle
```

Abnormal prompts：

```text
a photo of a damaged bottle
a photo of a defective bottle
a photo of an anomalous bottle
a photo of a broken bottle
```

得到两个文本原型：

```text
T_N：normal text prototype
T_A：abnormal text prototype
```

每个 patch 的文本异常分数：

```text
S_i_text = sim(F_i, T_A) - sim(F_i, T_N)
```

含义：

```text
S_i_text 越大，
表示该位置在 CLIP 语义空间里越像 abnormal，
而不是 normal。
```

这个分支提供：

```text
异常的正语义
```

这是 normal-only 模型发现未见异常的重要来源。

## 5. Normality Branch：负责“它像不像正常”

训练集全部是源域正常图像。

流程：

```text
source normal image
  -> frozen CLIP image encoder
  -> multi-level patch features
  -> lightweight residual adapter
  -> source normal patch memory / prototypes
```

建立 normal memory：

```text
M_N = {P_1^N, P_2^N, ..., P_K^N}
```

可以使用：

```text
clustering
coreset
prototype compression
```

测试 patch `F_i` 的 normality deviation：

```text
S_i_norm = min_{p in M_N} [1 - cos(F_i, p)]
```

如果能找到很像的正常 patch：

```text
S_i_norm ≈ 0
```

如果和正常模式差异很大：

```text
S_i_norm ↑
```

所以：

```text
Normality branch 负责发现“不像正常”的区域。
```

## 6. 两种异常证据融合

得到两个 patch-level 分数：

```text
S_i_text：语义上像不像异常
S_i_norm：结构上像不像正常
```

先基于 source-normal statistics 做 normalization，然后融合：

```text
A_i_raw = alpha * S_i_text + (1 - alpha) * S_i_norm
```

含义：

```text
一个区域最好同时满足：
1. 不像正常
2. 更符合 abnormal semantics

这样才得到更高异常响应。
```

这个设计比单独使用 PatchCore 或单独使用 WinCLIP 更合理。

核心贡献可以概括成：

```text
Semantic abnormality + Normality deviation
```

## 7. ARCC 的最终定位

ARCC 不负责：

```text
学习异常长什么样
从零产生 anomaly map
```

ARCC 只接受已有的：

```text
A_raw：原始异常图
F：CLIP patch context
```

然后输出 calibration map：

```text
(F, A_raw) -> ARCC -> G_cal
```

继续使用残差校准公式：

```text
A_final = A_raw + lambda * A_raw * tanh(G_cal)
```

也可以写成：

```text
A_final = A_raw * (1 + lambda * tanh(G_cal))
```

解释：

```text
Text + Normality 负责发现异常；
ARCC 负责判断异常响应是否具有上下文支持。
```

例如：

```text
孤立异常热点：
A_raw 高，但邻域和全局 context 都很正常，
G_cal < 0，
A_final 被降低，
从而抑制 false positive。
```

如果：

```text
一个区域本身高响应，
并且附近多个 patch 都出现结构异常，
G_cal > 0，
A_final 被保留或增强。
```

建议限制：

```text
0 < lambda < 1
```

这样 ARCC 只能校准原始异常响应，不能完全创造或推翻异常证据。

## 8. ARCC 在没有异常 GT 时怎么训练

这是本方案的关键点。

源域训练图全是正常图：

```text
Y(x_n) = 0
```

虽然没有 anomaly mask，但我们知道：

```text
正常图中的每个位置都应该低异常。
```

因此 ARCC 可以学习降低正常图上的 false positive：

```text
L_normal-map = Mean[sigma(A_final_normal)]
```

同时必须限制 ARCC 的修改幅度：

```text
L_cal = ||G_cal||
```

目的：

```text
防止 ARCC 永远输出负数，
把所有异常响应都消掉。
```

## 9. Normal-only 训练 Loss

第一版不要太复杂。

最终建议：

```text
L = L_text
  + lambda_1 * L_compact
  + lambda_2 * L_cons
  + lambda_3 * L_preserve
  + lambda_4 * L_normal-map
  + lambda_5 * L_cal
```

其中：

```text
L_text：
正常 patch 应该比 abnormal text 更接近 normal text。
```

公式：

```text
L_text = max(0, m + sim(F, T_A) - sim(F, T_N))
```

```text
L_compact：
正常 patch feature 应该更紧致。
```

```text
L_cons：
同一正常图不同 augmentation 的 anomaly map 应该一致。
```

```text
L_preserve：
限制 adapter 不要破坏 CLIP 原始特征。
```

公式：

```text
L_preserve = ||F_adapter - F_CLIP||_2
```

```text
L_normal-map：
正常图片上的异常响应应该低。
```

```text
L_cal：
限制 ARCC 不要过度修改原始 map。
```

整个训练过程不使用：

```text
anomaly image
anomaly mask
CutPaste
Perlin noise
synthetic lesion
synthetic anomaly
```

## 10. 测试阶段

测试时完全不训练。

目标图像：

```text
x_t
```

先得到 global anomaly score：

```text
S_g = sim(V_g, T_A) - sim(V_g, T_N)
```

同时 patch 分支：

```text
S_text + S_norm
  -> A_raw
  -> ARCC
  -> A_final
```

最终 pixel-level 输出：

```text
A_pixel = Upsample(A_final)
```

patch 图像级分数：

```text
S_p = MeanTopK(A_final)
```

最终 image score：

```text
S_image = S_g + beta * S_p
```

所以一个 forward 同时输出：

```text
image_score
anomaly_map
```

## 11. 如果需要 Binary Segmentation Mask

论文主指标直接使用 continuous heatmap 即可。

如果一定需要二值 mask，不要在 target GT 上找 threshold。

阈值应该从 source normal validation set 统计：

```text
tau = Q_99.5%(A_normal)
```

测试时：

```text
M(x) = 1[A_final > tau]
```

这样仍然没有使用 target GT。

## 12. 最终评价指标

Image-level：

```text
AUROC
AP
F1_max
```

Pixel-level：

```text
Pixel AUROC
Pixel AP
AUPRO
```

论文重点建议报告：

```text
Image AUROC
Image AP
Pixel AUROC
Pixel AP
AUPRO
```

由于没有异常 mask 训练，AUPRO 很重要，它可以证明区域定位质量。

## 13. Baseline 最终安排

建议分成两组比较。

| 类型 | 方法 | 训练异常 | Mask |
| --- | --- | --- | --- |
| Training-free | CLIP baseline | 不使用 | 不使用 |
| Training-free | WinCLIP | 不使用 | 不使用 |
| Normal-only | PatchCore cross-dataset | 不使用 | 不使用 |
| Normal-only | PaDiM cross-dataset | 不使用 | 不使用 |
| Ours | CLIP + Normality + ARCC | 不使用 | 不使用 |
| Strong supervision reference | AnomalyCLIP | 使用 | 使用 |
| Strong supervision reference | AA-CLIP | 使用 | 使用 |

注意：

```text
AnomalyCLIP / AA-CLIP 不属于完全公平的 supervision setting。
```

但建议保留，因为可以明确说明：

```text
它们获得了异常样本和 mask，
而我们只获得 source normal images。
```

如果性能接近，会很有说服力。

## 14. Ablation 必须做什么

最终 ablation chain：

```text
CLIP
  -> CLIP + Multi-level Patch
  -> + Text anomaly evidence
  -> + Normality deviation
  -> + ARCC
  -> + Global + Top-K Fusion
```

推荐表格：

| Variant | Text | Normality | Multi-level | ARCC | Image AUROC | Pixel AUROC | AUPRO |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP | yes | no | no | no |  |  |  |
| + Patch | yes | no | yes | no |  |  |  |
| + Normality | yes | yes | yes | no |  |  |  |
| + ARCC | yes | yes | yes | yes |  |  |  |
| Full | yes | yes | yes | yes |  |  |  |

最重要的对比：

```text
A_raw vs A_final
```

用来证明 ARCC：

```text
降低 normal false positive
提升 AUPRO
让热力图更加连续
不破坏真实 anomaly response
```

## 15. 可视化实验

论文 Figure 建议画：

```text
Input
  -> Text anomaly map
  -> Normality deviation map
  -> Raw fused map
  -> ARCC calibration map
  -> Final anomaly map
  -> GT
```

建议挑三类样本：

```text
1. Text branch 出现 false positive，ARCC 抑制。
2. Normality branch 找到异常，但 text 较弱。
3. Text + Normality 都支持异常区域，ARCC 增强或保持。
```

这个 Figure 可以直观解释创新点。

## 16. 最终论文贡献点

贡献 1：

```text
提出一种只使用源域正常图像训练的 cross-dataset zero-shot anomaly detection and localization framework，
无需真实异常图像、异常标签或像素级异常 mask。
```

贡献 2：

```text
将 CLIP 的 abnormal semantic evidence 与 source-normal normality deviation 结合，
使未见异常能够通过“像异常”和“不像正常”两种互补证据被发现。
```

贡献 3：

```text
提出 ARCC context calibration，
对原始异常响应进行上下文感知残差校准，
在没有异常 mask 监督的情况下抑制正常区域假阳性并改善异常区域定位。
```

## 17. 代码执行顺序

新主线文件：

```text
/Users/Admin/PyCharmMiscProject/train.py
/Users/Admin/PyCharmMiscProject/model/clip_ad.py
/Users/Admin/PyCharmMiscProject/trainer/clip_ad_trainer.py
/Users/Admin/PyCharmMiscProject/configs/clip_ad/clip_ad_medical.py
```

建议新增模块：

```text
/Users/Admin/PyCharmMiscProject/model/modules/adapters.py
/Users/Admin/PyCharmMiscProject/model/modules/prompt_templates.py
/Users/Admin/PyCharmMiscProject/model/modules/normal_memory.py
/Users/Admin/PyCharmMiscProject/model/modules/arcc.py
/Users/Admin/PyCharmMiscProject/model/modules/scoring.py
```

第一轮代码目标：

```text
CLIP baseline
  -> Text anomaly map
  -> Normality deviation map
  -> Raw fused map
  -> Image score
```

第二轮代码目标：

```text
+ ARCC calibration
  -> Final anomaly map
  -> A_raw vs A_final ablation
```

第三轮代码目标：

```text
+ 多层 patch feature
+ normal memory 压缩
+ source normal validation threshold
+ 完整 cross-dataset evaluation
```

最先做的实验：

```text
1. CLIP baseline
2. Text + Normality
3. Text + Normality + ARCC
```

先验证这三级是否持续提升，再决定是否增加其他模块。

## 18. 按阶段添加模块路线

后续代码不要一次性把所有模块都写进去，而是按照下面顺序逐步添加。每一阶段都必须满足：

```text
能单独运行
能输出 image_score
能输出 anomaly_map
能保存结果
能和上一阶段做 ablation
```

### Stage 1：CLIP Text-only Baseline

目标：

```text
先只用 frozen CLIP + fixed text prompts，
得到最基础的图像级异常分数和 patch 文本异常热力图。
```

包含模块：

```text
Frozen CLIP image encoder
Frozen CLIP text encoder
Prompt templates
Text anomaly scorer
Top-K image scorer
```

暂不加入：

```text
Normal memory
Adapter
ARCC
Synthetic anomaly
异常图训练
异常 mask 训练
```

输出：

```text
S_global
S_text_map
A_pixel = upsample(S_text_map)
S_image = S_global + beta * TopK(A_pixel)
```

成功标准：

```text
在目标数据集 test split 上可以直接评价 image-level 和 pixel-level 指标。
```

### Stage 2-pre：Multi-level Raw CLIP Text Map

目标：

```text
不只使用最后一层 patch feature，
而是提取多层 CLIP patch features，
直接做 fixed text similarity map，
验证 raw CLIP multi-level patch map 是否能提升定位。
```

新增模块：

```text
Multi-level patch extractor
Patch projection / normalization
Multi-level map fusion
```

关键链路：

```text
frozen CLIP image encoder
  -> multi-level raw patch tokens
  -> projection / normalization
  -> sim(T_A) - sim(T_N)
  -> mean-fused text anomaly map
```

暂不加入：

```text
Adapter
Normal memory
CSSD / Mamba
ARCC
Synthetic anomaly
```

输出：

```text
S_text_map_multi
A_pixel
S_image
```

消融对比：

```text
Stage 1：single-level patch
Stage 2-pre：multi-level raw patch
```

实验结论：

```text
结果明显变差，说明简单 multi-level raw CLIP text map 会冲淡最后层语义响应，
不适合作为主路线，只保留为负向消融。
```

### Stage 2A：CLIP + Lightweight Patch Adapter

目标：

```text
验证 normal-only patch refinement 是否有效：
raw CLIP patch token 先经过一个轻量残差 MLP adapter，
得到 refined patch token，
再和 fixed normal / abnormal text prototype 做相似度打分。
```

关键链路：

```text
image
  -> frozen CLIP image encoder
  -> raw CLIP patch tokens
  -> Residual MLP patch_adapter
  -> refined patch tokens
  -> sim(T_A) - sim(T_N)
  -> anomaly_map
```

训练对象：

```text
只训练 Residual MLP patch_adapter
冻结 CLIP image encoder
冻结 CLIP text encoder
固定 prompt templates
不训练 text prototype
```

训练数据：

```text
MVTec-AD train/good normal images only
```

训练 loss：

```text
patch_score = sim(refined_patch, T_A) - sim(refined_patch, T_N)

L_normal_topk = softplus(TopK(patch_score) + margin)
L_consistency = 1 - cosine(refined_patch, raw_patch)
L_image_normal = BCEWithLogits(image_score, 0)

L = L_normal_topk
  + 0.1 * L_consistency
  + 0.1 * L_image_normal
```

输出：

```text
S_text_map_refined
A_pixel
S_image
```

消融定位：

```text
Stage 1：raw CLIP patch text-only
Stage 2-pre：multi-level raw CLIP text-only
Stage 2A：raw CLIP patch -> MLP adapter -> refined patch -> text score
```

实验结论：

```text
Stage 2A 显著提升 pixel-level 指标，
证明 raw CLIP patch 经过 normal-only refinement 后，
再做 fixed text similarity 是有效的。
```

### Stage 2B：CLIP + Multi-layer CNN Residual Patch Adapter

目标：

```text
保留 CLIP 最后一层 patch token 作为 text-aligned semantic base；
用 CLIP 中低层 patch tokens 提供局部纹理和边界信息；
CNN 只产生 local_delta，
通过 AA-CLIP 风格 residual update 修正最后层 patch token。
```

关键链路：

```text
image
  -> frozen CLIP image encoder
  -> layer 12 / 18 / 24 patch tokens
  -> CNN(layer 12, layer 18) = local_delta
  -> refined_patch = normalize(layer24_patch + alpha * local_delta)
  -> A_raw = sim(refined_patch, T_A) - sim(refined_patch, T_N)
  -> G_mamba = Mamba(last_patch)
  -> A_final = ARCC(A_raw, G_mamba, refined_patch)
```

分支动机：

```text
CNN local branch：
使用 layer 12 / 18 的 patch tokens，
补充最后层 CLIP patch token 缺少的局部空间归纳偏置，
增强纹理变化、边缘破坏、缺陷边界和局部连续性。

Layer 24 semantic base：
保留 CLIP 最后一层的 text alignment，
确保最终 anomaly map 仍由 refined patch 与 normal/abnormal text prototypes 计算得到。

注意：
Stage 2B 不让 CNN 直接输出 heatmap。
Stage 2B 不让 Mamba/CSSD 和 CNN 分别输出 heatmap 再相加。
Stage 2B 不让 text prompt 进入 CNN。
Stage 2B 只做 feature-level residual refinement。
Mamba 只做 A_raw 后的 global response context modeling。
ARCC 是唯一直接输出 A_final 的 map-level response calibration。
```

代码对应关系：

```text
MAMBAADZeroShot 的 v_raw
= CLIP Stage 2B 的 layer24_patch_feat

AA-CLIP adapter residual update
= CLIP Stage 2B 的 normalize(layer24_patch + alpha * local_delta)

MAMBAADZeroShot 的 v_refined
= CLIP Stage 2B 的 refined_patch_feat

MAMBAADZeroShot 的 anomaly_map
= refined_patch_feat 和 CLIP 文本原型比较得到的 map
```

训练对象：

```text
只训练 multi-layer CNN patch_adapter 和 ARCC
冻结 CLIP image encoder
冻结 CLIP text encoder
固定 prompt templates
不使用 synthetic anomaly
当前 supervised 主线使用 source-domain abnormal masks
```

训练 loss：

```text
当前 supervised source-mask loss：

L_mask_bce(A_final, mask)
L_mask_dice(A_final, mask)
L_raw_bce(A_raw, mask)
L_image_supervised(image_score, label)
L_normal_topk
L_consistency(refined_patch, layer24_patch)
L_arcc_cal
```

成功标准：

```text
Cross-category：
source categories with masks -> held-out target category test，
确认没有目标类别样本进入训练。

A_raw vs A_final：
验证 ARCC 是否在唯一 map-level refinement 位置提升 AUPRO / Pixel AP / Pixel F1。

Layer ablation：
CNN local 使用 layer 12/18 对比只用 layer 18，
验证中低层局部纹理是否优于只用第 24 层。
```

当前 Stage 2B 主线：

```text
CLIP layer 12 / 18 -> CNN local_delta
CLIP layer 24 -> text-aligned semantic base
refined_patch = normalize(layer24 + alpha * local_delta)
A_raw = sim(refined_patch, T_abnormal) - sim(refined_patch, T_normal)
G_mamba = Mamba(last_patch)
A_final = ARCC(A_raw, G_mamba, refined_patch)
```

### Stage 1 / Stage 2-pre / Stage 2A / Stage 2B 已跑结果对比

当前 MVTec-AD test split 结果如下，数值均为百分制 Avg 行。
这些结果来自旧的 normal-only / no-mask-supervision 路线，
现在只作为结构消融参考；新的主实验应改看
`configs/clip_ad/clip_ad_supervised_mask_full.py`。

| Stage | 结构 | Image AUROC | Image AP | Image F1-max | AUPRO | Pixel AUROC | Pixel AP | Pixel F1-max | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Stage 1 | single-level raw CLIP patch -> fixed text score | 86.448 | 94.215 | 89.375 | 7.760 | 35.984 | 3.156 | 6.732 | image-level 可用，pixel-level 弱 |
| Stage 2-pre | multi-level raw CLIP patch -> fixed text score -> mean fusion | 85.997 | 94.020 | 89.620 | 2.627 | 21.804 | 2.303 | 6.213 | 定位明显变差，保留为负向消融 |
| Stage 2A | raw CLIP patch -> MLP adapter -> refined patch -> fixed text score | 88.991 | 95.002 | 91.482 | 15.359 | 52.878 | 7.003 | 12.174 | 明显提升，证明 normal-only patch refinement 有效 |
| Stage 2B-1 | raw CLIP patch -> CSSD / Mamba scan-only -> refined patch -> fixed text score | 86.454 | 94.282 | 90.118 | 9.403 | 38.665 | 4.620 | 9.810 | 超过 Stage 1，但弱于 Stage 2A；说明全局上下文有帮助但不能单独替代局部 refinement |
| Stage 2B-2 | raw CLIP patch -> CNN local + Mamba global -> refined patch -> fixed text score | 86.209 | 94.164 | 89.954 | 11.009 | 39.529 | 4.697 | 9.933 | 最新复跑低于 Stage 2A；说明当前 local + global 直接融合还不稳定 |
| Stage 2B-2 rerun | raw CLIP patch -> CNN local + Mamba global -> refined patch -> fixed text score | 86.223 | 94.158 | 89.985 | 10.640 | 39.353 | 4.695 | 9.611 | 复跑趋势一致，当前版本不能作为最强结构 |
| Stage 2B-3 | CSSD / Mamba refined patch -> text score A_raw -> ARCC -> A_final | 86.709 | 94.334 | 90.143 | 13.295 | 47.806 | 5.634 | 11.376 | ARCC 后优于 Stage 2B-2，但仍弱于 Stage 2A + ARCC |
| Stage 2C | Stage 2A raw map -> ARCC response calibration -> final map | 89.243 | 95.050 | 91.694 | 17.469 | 55.094 | 7.797 | 12.967 | 旧路线里当前最好，证明 ARCC 对 MLP adapter 有帮助 |
| Full | CNN local + Mamba global refined patch -> text score A_raw -> ARCC -> A_final | TBD | TBD | TBD | TBD | TBD | TBD | TBD | 待跑，完整方法 |

相对 Stage 1，Stage 2A 的 Avg 提升：

```text
Image AUROC: +2.543
Image AP:    +0.787
AUPRO:       +7.599
Pixel AUROC: +16.894
Pixel AP:    +3.847
Pixel F1:    +5.442
```

Stage 2A 当前建议 checkpoint：

```text
优先使用 epoch 5 / net_5.pth，
因为 Avg AUPRO、Pixel AUROC、Pixel F1 都在 epoch 5 达到当前最好。
Pixel AP 的 Max 为 7.105，出现在 epoch 3，但优势很小。
```

最新旧路线结果的 Max 诊断：

```text
Stage 2C / Stage 2A + ARCC:
  Max AUPRO 18.507, Max Pixel AUROC 56.556, Max Pixel AP 7.970, Max Pixel F1 13.651.

Stage 2B-3 / CSSD + ARCC:
  Max AUPRO 18.269, Max Pixel AUROC 60.072, Max Pixel AP 8.736, Max Pixel F1 14.025.

Stage 2B-2 / local-global latest rerun:
  Max Pixel AUROC 52.251, Max Pixel AP 6.662, Max Pixel F1 11.578.
```

解释：

```text
如果只是判断结构有没有潜力，可以看 Max。
如果要写正式表格，不能用 target test 的 Max 直接挑 checkpoint；
应使用固定 epoch、source validation，或预先声明 checkpoint selection rule。
```

### Stage 2C：Stage 2A + ARCC Response Calibration

目标：

```text
在当前最强的 Stage 2A MLP adapter baseline 上接入 ARCC，
比较 A_raw 和 A_final，
验证 response-level context calibration 是否能进一步降低 false positive。
```

关键链路：

```text
raw CLIP patch
  -> MLP patch adapter
  -> refined patch token
  -> fixed text similarity
  -> A_raw
  -> ARCCCalibration
  -> A_final
```

ARCC 输入：

```text
feature_map：refined patch tokens reshape 得到的 24 x 24 特征图
local_logits：A_raw 的 patch-level text similarity map
```

训练 loss：

```text
沿用 Stage 2A：
L_normal_topk
L_consistency
L_image_normal

新增：
L_arcc_normal = BCEWithLogits(A_final_normal, 0)
L_arcc_cal = ||G_cal||^2
```

实验重点：

```text
A_raw vs A_final
Stage 2A vs Stage 2A + ARCC
AUPRO / Pixel AUROC / Pixel AP / normal false positive heatmap
```

### Stage 3：加入 Normal Patch Memory

目标：

```text
只用 source normal images 建立正常 patch memory，
让模型知道“什么区域不像正常结构”。
```

新增模块：

```text
NormalMemory
Memory builder
Patch distance scorer
Source-normal score normalization
```

训练 / 建库数据：

```text
MVTec-AD train/good
```

测试数据：

```text
VisA / BTAD / MPDD test
```

输出：

```text
S_norm_map
```

消融对比：

```text
Stage 2：Text anomaly only
Stage 3：Text anomaly + Normality deviation
```

成功标准：

```text
Normality deviation 能补充 text branch 找不到的异常区域。
```

### Stage 4：Text + Normality 融合

目标：

```text
融合“像不像异常”和“像不像正常”两种证据，
得到 raw anomaly map。
```

新增模块：

```text
Score normalizer
Raw map fusion
Alpha fusion weight
```

公式：

```text
A_raw = alpha * S_text + (1 - alpha) * S_norm
```

输出：

```text
A_raw
S_patch = MeanTopK(A_raw)
S_image = S_global + beta * S_patch
```

消融对比：

```text
Text only
Normality only
Text + Normality
```

成功标准：

```text
融合后 image-level 和 pixel-level 指标整体优于单分支。
```

### Stage 5：加入 Lightweight Adapter

目标：

```text
只在 source normal images 上轻量适配 patch feature，
但不破坏 CLIP 原始语义空间。
```

新增模块：

```text
ResidualAdapter
Adapter preserve loss
Normal compactness loss
```

训练数据：

```text
source normal images only
```

禁止：

```text
source anomaly images
source anomaly masks
target images
target masks
synthetic anomaly
```

核心 loss：

```text
L_preserve = ||F_adapter - F_CLIP||_2
L_compact
L_text
```

消融对比：

```text
without adapter
with adapter
```

成功标准：

```text
Adapter 提升 normality deviation 的稳定性，同时不明显降低 text branch 的语义判断。
```

### Stage 6：加入 ARCC Calibration

目标：

```text
ARCC 不从零生成异常图，
只对 A_raw 做上下文残差校准。
```

新增模块：

```text
ARCC
Calibration map G_cal
Residual calibration formula
```

公式：

```text
A_final = A_raw * (1 + lambda * tanh(G_cal))
```

训练约束：

```text
L_normal-map = Mean[sigma(A_final_normal)]
L_cal = ||G_cal||
```

输出：

```text
G_cal
A_final
```

消融对比：

```text
A_raw vs A_final
without ARCC vs with ARCC
```

成功标准：

```text
ARCC 能降低 normal false positive，
提升 AUPRO，
并让热力图更连续。
```

### Stage 7：完整 Cross-dataset Evaluation

目标：

```text
固定模型与超参数，
完成跨数据集目标域 zero-shot 测试。
```

实验协议：

```text
MVTec normal-only -> VisA
MVTec normal-only -> BTAD
MVTec normal-only -> MPDD
VisA normal-only -> MVTec-AD
```

输出结果：

```text
Image AUROC
Image AP
F1_max
Pixel AUROC
Pixel AP
AUPRO
```

成功标准：

```text
结果表、消融表、可视化图都能支持最终方法叙事。
```

### Stage 8：论文图和可视化

目标：

```text
把代码输出变成论文中可解释的图。
```

需要保存：

```text
Input
Text anomaly map
Normality deviation map
Raw fused map
ARCC calibration map
Final anomaly map
GT mask
```

最终图示重点：

```text
Text branch 提供 abnormal semantic evidence
Normality branch 提供 source-normal deviation
ARCC 抑制 false positive 并校准上下文
```
