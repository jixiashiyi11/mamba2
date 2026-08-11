# HANDOFF

更新时间：2026-07-18

这份文档写给一个完全没有上下文的新会话。当前项目是医学零样本/normal-only 异常检测与定位，主要围绕 brain、liver、retinal，同时额外测试过 BUSI breast、MSD Liver、BraTS、retinal lesion benchmark。

## 1. 当前任务到底在做什么

用户想做一篇医学异常检测/定位方向的论文。核心设定大致是：

```text
不使用目标测试集异常样本训练
主要使用 auxiliary normal-only medical images
测试 brain / liver / retinal 等医学异常定位任务
```

当前最稳的主线不是 AnomalyCLIP 融合，也不是纯 BiomedCLIP prompt。主线是：

```text
Frozen BiomedCLIP feature
→ CNN local adapter 生成局部异常响应 A_local
→ ARCC 进行上下文校准
→ 输出 anomaly map
→ 同时评估 image-level 和 pixel-level 指标
```

ARCC 的完整含义：

```text
Anomaly-Response-guided Context Calibration
```

但目前表现最好的版本其实是 ARCC E2，即 feature-based calibration，不输入 response / foreground / edge prior。

## 2. 当前最重要的 baseline / 主结果

目前最推荐作为主方法结果的 run 是：

```text
runs/MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333
```

对应配置：

```text
configs/mambaad/a_arcc_e2_feature_calib_e5.py
```

checkpoint：

```text
net.pth
```

结果：

| Organ | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | F1-max |
|---|---:|---:|---:|---:|---:|
| brain | 83.243 | 94.220 | 76.657 | 19.565 | 27.616 |
| liver | 57.769 | 96.078 | 86.015 | 5.034 | 8.963 |
| retinal | 79.745 | 89.176 | 60.833 | 24.191 | 32.288 |
| Avg | 73.585 | 93.158 | 74.502 | 16.263 | 22.956 |

注意：用户经常会记得 brain 有 89 的结果。那个 89 是另一个 OASIS brain normal 实验，不是 ARCC E2 的 brain 数值。

## 3. OASIS brain 相关结果

有一个加入 OASIS brain normal 后的实验：

```text
runs/MAMBAADZeroShotTrainer_configs_mambaad_a_cnn_global_aux_oasisbrain_e5_20260707-115929
```

对应 run config：

```text
a_cnn_global_aux_oasisbrain_e5.py
```

它不是 ARCC。它是：

```text
BiomedCLIP + CNN local adapter + Mamba/global aux
```

结果：

| Organ | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | F1-max |
|---|---:|---:|---:|---:|---:|
| brain | 89.79 | 92.22 | 70.29 | 12.67 | 18.85 |
| liver | 52.19 | 96.16 | 84.81 | 4.95 | 9.83 |
| retinal | 84.88 | 84.91 | 54.03 | 20.66 | 28.11 |
| Avg | 75.62 | 91.10 | 69.71 | 12.76 | 18.93 |

解释：

```text
OASIS brain normal 明显提升 brain image AUROC，
但整体 pixel localization 不如 ARCC E2。
```

还有一个容易混淆点：ARCC E2 的配置本身不一定显式写 OASIS，但它训练 root 是：

```text
data/medical_aux_train_balanced/train/good
```

服务器上这个目录里已经有：

```text
data/medical_aux_train_balanced/train/good/oasis_brain
```

所以如果当时训练时该目录已经存在，ARCC E2 很可能间接使用了 OASIS normal brain。后续写论文时要谨慎表述，不要说“完全未见 brain normal”。

## 4. ARCC 到底是什么

通俗解释：

```text
CNN 先判断哪里可疑，得到 A_local。
ARCC 不重新找异常，而是围绕这些局部响应采样上下文，
生成一个 calibration map，
最后增强可信异常、压低不可信响应。
```

公式：

```text
A_final = A_local + lambda * A_local * tanh(G_cal)
```

有两个 deformable 概念，别混：

| 名称 | 位置 | 作用 |
|---|---|---|
| CSSD/Mamba deformable pooling | Mamba/CSSD feature branch 内部 | 做 feature/context refinement |
| ARCC deformable calibration | CNN 输出 A_local 之后 | 校准 anomaly map |

不要把 ARCC 说成“就是 Mamba 里的 deformable CNN”。更准确是：

```text
CNN 负责局部异常证据；
Mamba/CSSD 或 feature branch 提供上下文；
ARCC 用 deformable context sampling 校准局部响应。
```

## 5. 外部 benchmark 已跑结果

### BUSI Breast

用 E2 checkpoint 测 BUSI：

| Organ | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | F1-max |
|---|---:|---:|---:|---:|---:|
| breast | 81.000 | 75.704 | 48.704 | 23.625 | 27.519 |

结论：

```text
image-level 泛化还可以，但 pixel localization 不强。
可以作为外部 benchmark，不适合作为最强主结果。
```

### MSD Liver Standard Test

E3 checkpoint 在 MSD Liver 标准测试集上：

| Organ | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | F1-max |
|---|---:|---:|---:|---:|---:|
| liver | 68.580 | 91.658 | 71.346 | 5.641 | 11.638 |

结论：

```text
比原 liver test 的 image AUROC 好，但 Pixel AP 仍然低。
```

### BraTS Brain Standard Test

E3 checkpoint 在 BraTS 上：

| Organ | Image AUROC | Pixel AUROC | AUPRO | Pixel AP | F1-max |
|---|---:|---:|---:|---:|---:|
| brain | 63.443 | 96.522 | 78.842 | 26.474 | 31.680 |

结论：

```text
Pixel AUROC 很高，但 image AUROC 很低。
再次说明 image-level score 和 pixel-level map 不一致。
```

## 6. 频率分析已经完成

我们基于带 mask 的医学 benchmark 做了 lesion frequency 分析。

输出路径：

```text
outputs/lesion_frequency_analysis
```

统计结果：

| Organ | Low | Mid | High | 结论 |
|---|---:|---:|---:|---|
| brain | 0.6087 | 0.2226 | 0.1687 | 明显低频主导 |
| liver | 0.3202 | 0.2998 | 0.3800 | 中高频更多 |
| retinal | 0.3001 | 0.3059 | 0.3940 | 中高频更多 |

PPT 可以写：

```text
Lesion frequency is organ-dependent: brain lesions are mainly low-frequency, while liver and retinal lesions retain more mid/high-frequency information. This motivates organ-aware frequency modeling.
```

中文：

```text
病灶频率具有器官差异：脑部病灶更偏低频，肝脏和视网膜病灶保留更多中高频信息。
```

注意：这个分析适合做 motivation，不代表频域方法已经成功。

## 7. 已经尝试但不建议押主线的方向

### 7.1 Frequency Dual-Role Synthetic

思路：

```text
positive frequency lesion: 前景内频域伪病灶，要亮
negative frequency nuisance: 背景/边缘频域干扰，要暗
```

代表 run：

```text
runs/MAMBAADZeroShotTrainer_configs_mambaad_a_cnn_global_aux_frequency_dual_role_e5_20260715-111536
```

结果：

| Version | Avg Image AUROC | Avg Pixel AUROC | Avg AUPRO | Pixel AP | F1-max |
|---|---:|---:|---:|---:|---:|
| ARCC E2 | 73.585 | 93.158 | 74.502 | 16.263 | 22.956 |
| Frequency dual-role e5 | 75.194 | 84.071 | 46.207 | 6.033 | 10.841 |

结论：

```text
image AUROC 接近或略好，但 localization 明显坏。
不建议作为主线。
```

### 7.2 Normality Explainer

思路：

```text
用 normal prototype / normality explainability 压低 raw map。
```

结果：

```text
raw pixel_AUROC = 0.9437
gamma0.5       = 0.9420
gamma1         = 0.9297
gamma2         = 0.9110
```

结论：

```text
直接 normality suppression 会误伤真实病灶。
不要做主线。
```

### 7.3 Region Verification

思路：

```text
先从 heatmap 提 connected components，
再用 shape / score-drop 等方式验证候选区域。
```

结果：

```text
component_only: image_AUROC=0.7315 pixel_AUROC=0.4835 pixel_AP=0.0444 F1=0.1149
raw: image_AUROC=0.7397 pixel_AUROC=0.8544 pixel_AP=0.1055 F1=0.1585
region_verified: image_AUROC=0.7245 pixel_AUROC=0.5339 pixel_AP=0.0454 F1=0.1123
soft_region_verified: image_AUROC=0.7419 pixel_AUROC=0.8537 pixel_AP=0.1053 F1=0.1586
```

结论：

```text
硬 region verification 会破坏 pixel metric。
soft 版本几乎等于 raw，只略微提升 image AUROC。
不适合当大贡献。
```

### 7.4 Pseudo-Anomaly Memory / PA-CLIP Inspired

思路：

```text
借鉴 PA-CLIP，用 memory / pseudo-anomaly 抑制假阳性。
```

结果：

```text
raw: image_AUROC=0.7397 pixel_AUROC=0.8495 pixel_AP=0.0818 F1=0.1352
pseudo_veto_l0.5: image_AUROC=0.7388 pixel_AUROC=0.9051 pixel_AP=0.1059 F1=0.1629
```

解释：

```text
pseudo_veto_l0.5 对 pixel 有改善，但 image 没提升，且它偏 eval/post-hoc。
可以作为分析或补充实验，不建议取代 ARCC 主线。
```

### 7.5 FPC: Frequency Perturbation Consistency

FPC 不是 synthetic blob。它不会在某个固定位置生成圆形异常，而是在 test/eval 时对整张图做局部频率扰动，检查响应是否稳定。

近期结果：

```text
fpc_direct: image_AUROC=0.5004 pixel_AUROC=0.8895 pixel_AP=0.1025 F1=0.1851
fpc_soft_l0.25: image_AUROC=0.7401 pixel_AUROC=0.8467 pixel_AP=0.0812 F1=0.1353
fpc_soft_l0.5: image_AUROC=0.7403 pixel_AUROC=0.8438 pixel_AP=0.0806 F1=0.1354
fpc_soft_l0.75: image_AUROC=0.7404 pixel_AUROC=0.8409 pixel_AP=0.0800 F1=0.1355
raw: image_AUROC=0.7397 pixel_AUROC=0.8495 pixel_AP=0.0818 F1=0.1352
raw_positive: image_AUROC=0.5089 pixel_AUROC=0.9211 pixel_AP=0.1358 F1=0.2335
```

结论：

```text
FPC 说明频域里确实有 lesion-related signal，
但直接融合不稳定，尤其 image AUROC 会崩。
因此 FPC 适合作 diagnostic，不适合做主方法。
```

## 8. 最近关于 retinal 背景红的问题

用户上传了一张 OCT/retinal 可视化，背景看起来也有红色。已经检查本地 CSV：

```text
runs/MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333/debug_eval_normal_response_calib/foreground_mask_diagnostic.csv
runs/MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333/debug_eval_normal_response_calib/false_positive_region_diagnostic.csv
runs/MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333/debug_eval_busi/foreground_mask_diagnostic.csv
runs/MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333/debug_eval_busi/false_positive_region_diagnostic.csv
```

关键 retinal 诊断：

```text
mean_foreground_ratio = 0.9735
mean_topk_background_fraction = 0.0201
mean_topk_foreground_edge_fraction = 0.0083
mean_topk_foreground_interior_fraction = 0.9717
```

解释：

```text
retinal 的 top-k 高响应并不是主要落在真正 background；
只有约 2% 在 background。
图上背景红主要是 foreground mask 太宽 + OCT 背景 speckle + 单图 min-max 归一化造成的视觉现象。
```

后续如果要修：

```text
先只改 visualization，不改 metric。
retinal 可视化时用更严格 foreground threshold，例如 0.08 或 0.10。
foreground 外强制置为 map 最低值/蓝色。
```

不要立刻说模型真的在背景强误报，CSV 更支持“可视化和 foreground mask 问题”。

## 9. 当前最可能继续发展的新主线

最近给用户建议的新主线是：

```text
Self-Anchored Normality Calibration
自锚定正常性校准
```

核心思想：

```text
不要用外部 normal bank 直接压制，也不要只依赖 synthetic lesion。
每张测试图自己提供 normal anchors：
在 foreground 内选择 raw anomaly map 低响应 patch 作为本图正常锚点。
再判断高响应 patch 是否真的偏离这些同图正常锚点。
```

流程：

```text
Input image
→ BiomedCLIP patch features
→ E2 ARCC raw map A_raw
→ 取 foreground 内 A_raw 最低的若干 patch 作为 self normal anchors
→ 计算每个 patch 与 anchors 的相似度 normality
→ 得到 unexplained_score = 1 - normality
→ 用 unexplained_score 校准 A_raw
→ A_final
```

公式可先用：

```text
N_i = max cosine similarity(patch_i, self_normal_anchors)
unexplained_i = 1 - N_i
A_final_i = A_raw_i * (1 + lambda * unexplained_i)
```

它和之前 normal bank 的区别：

```text
以前 normal bank：用训练集 normal patches 解释测试图，跨器官/跨模态容易失败。
Self-anchor：每张图从自己内部取正常锚点，适配当前图像的模态、亮度和结构。
```

它和 ARCC 的关系：

```text
ARCC 找可疑区域；
Self-anchor 判断这个可疑区域是否偏离本图自己的正常结构。
```

这条线和 E2/ARCC 最自然，不像 FPC 那样割裂。建议下一步先做 eval-only，不要重训。

建议最小实验：

```text
E0: ARCC E2 raw map
E1: raw + self-anchor calibration
E2: raw + self-anchor + foreground/edge optional filtering
E3: raw + self-anchor + organ frequency diagnostic only
```

先看：

```text
Pixel AP
F1-max
AUPRO
top1-hit
top5-hit
normal image high-response ratio
false_positive_region_diagnostic
```

如果 Pixel AP/F1 提升而 Pixel AUROC 不明显下降，这个方向值得推进。

## 10. 当前代码状态

工作区是 dirty 的。最近 `git status --short` 显示：

```text
 M model/mambaad.py
 M trainer/mambaad_trainer.py
?? HANDOFF.md
?? configs/mambaad/a_arcc_e2_busi_net_test.py
?? configs/mambaad/a_arcc_e2_busi_test.py
?? configs/mambaad/a_arcc_e2_freq_role_e5.py
?? configs/mambaad/a_arcc_e2_freqaux_auxonly_e5.py
?? configs/mambaad/a_arcc_e2_freqaware_e1_freqonly_e5.py
?? configs/mambaad/a_arcc_e2_freqaware_e2_fixed_e5.py
?? configs/mambaad/a_arcc_e2_freqaware_e3_gated_e5.py
?? configs/mambaad/a_arcc_e2_freqaware_e4_residual_gate_e5.py
?? configs/mambaad/a_arcc_e2_half_synth_e5.py
?? configs/mambaad/a_arcc_e2_highfreq_detail_e5.py
?? configs/mambaad/a_arcc_e2_normal_response_calib_test.py
?? configs/mambaad/a_arcc_e3_busi_test.py
?? configs/mambaad/a_arcc_e3_half_synth_e5.py
?? configs/mambaad/a_arcc_e3_retinal_lesion_test.py
?? configs/mambaad/a_cnn_global_aux_freq_*.py
?? figures/arcc_overview_editable.pptx
?? outputs/
?? third_party/
?? tools/analyze_lesion_frequency.py
?? tools/create_arcc_overview_pptx.py
?? tools/deletion_consistency_eval.py
?? tools/frequency_map_consistency_eval.py
?? tools/frequency_perturbation_consistency_eval.py
?? tools/normal_response_calibration_eval.py
?? tools/normality_explainer_suppression_eval.py
?? tools/prepare_busi_ad_test.py
?? tools/prepare_retinal_lesion_ad_test.py
?? tools/pseudo_anomaly_memory_eval.py
?? tools/region_verified_eval.py
```

注意：

```text
当前本地代码未必等于干净 ARCC E2 baseline。
如果用户要求“回到最好的版本”，优先参考 commit:
31c7a632 Add ARCC calibration and benchmark configs
```

不要随便 `git reset --hard`，因为有很多用户需要的实验脚本是未提交文件。需要回退时，应先问用户是否保留未提交脚本，或者只 checkout 指定文件。

## 11. 重要脚本和配置

有用脚本：

```text
tools/analyze_lesion_frequency.py
tools/deletion_consistency_eval.py
tools/frequency_perturbation_consistency_eval.py
tools/normality_explainer_suppression_eval.py
tools/pseudo_anomaly_memory_eval.py
tools/region_verified_eval.py
tools/prepare_busi_ad_test.py
tools/prepare_retinal_lesion_ad_test.py
```

重要配置：

```text
configs/mambaad/a_arcc_e2_feature_calib_e5.py
configs/mambaad/a_arcc_e3_response_calib_e5.py
configs/mambaad/a_arcc_e4_full_calib_e5.py
configs/mambaad/a_arcc_e2_busi_test.py
configs/mambaad/a_arcc_e3_brats_test.py
configs/mambaad/a_arcc_e3_msd_liver_test.py
configs/mambaad/a_arcc_e3_retinal_lesion_test.py
```

## 12. 常用服务器命令

### CLIP Stage2C / ARCC 当前需要同步的文件

本轮 CLIP normal-only 线路修改过、或者运行 Stage2C 必须同步的文件：

```text
train.py
configs/__init__.py
data/folder_ad.py
model/clip_ad.py
model/mambaad.py
model/modules/
trainer/clip_ad_trainer.py
configs/clip_ad/
tools/build_supervised_meta.py
plan/20260807-114546-clip-modular-redesign-plan.md
HANDOFF.md
```

关键新增 / 修改点：

```text
configs/clip_ad/clip_ad_mtvecad_stage2a_arcc.py
configs/clip_ad/clip_ad_mtvecad_stage2b_cssd_arcc.py
configs/clip_ad/clip_ad_mtvecad_stage2b_local_global_arcc.py
configs/clip_ad/clip_ad_supervised_mask_full.py
tools/build_supervised_meta.py
configs/clip_ad/clip_ad_mtvecad_stage2b_local_global.py
configs/clip_ad/clip_ad_mtvecad_stage2b_cssd.py
model/modules/adapters.py
model/clip_ad.py
trainer/clip_ad_trainer.py
model/mambaad.py
```

### 推荐一次性上传命令

这个命令会同步当前 CLIP Stage1 / Stage2-pre / Stage2A / Stage2B / Stage2C 相关代码。
注意必须包含 `model/mambaad.py`，因为 CSSD / ARCC 都复用这里的代码。

```bash
cd /Users/Admin/PyCharmMiscProject

rsync -avP --relative -e "ssh -p 2222" \
  ./train.py \
  ./configs/__init__.py \
  ./model/clip_ad.py \
  ./model/mambaad.py \
  ./model/modules/ \
  ./trainer/clip_ad_trainer.py \
  ./configs/clip_ad/ \
  ./data/folder_ad.py \
  ./tools/build_supervised_meta.py \
  ./plan/20260807-114546-clip-modular-redesign-plan.md \
  ./HANDOFF.md \
  wmwanghkmu@localhost:/home/wmwanghkmu/ZYH/Mamba/ADer/
```

### 分组上传命令

如果只想上传代码：

```bash
cd /Users/Admin/PyCharmMiscProject

rsync -avP --relative -e "ssh -p 2222" \
  ./train.py \
  ./configs/__init__.py \
  ./model/clip_ad.py \
  ./model/mambaad.py \
  ./model/modules/ \
  ./trainer/clip_ad_trainer.py \
  ./configs/clip_ad/ \
  ./data/folder_ad.py \
  ./tools/build_supervised_meta.py \
  wmwanghkmu@localhost:/home/wmwanghkmu/ZYH/Mamba/ADer/
```

如果只上传 Stage2C 新配置：

```bash
cd /Users/Admin/PyCharmMiscProject

rsync -avP -e "ssh -p 2222" \
  configs/clip_ad/clip_ad_mtvecad_stage2a_arcc.py \
  wmwanghkmu@localhost:/home/wmwanghkmu/ZYH/Mamba/ADer/configs/clip_ad/
```

如果只上传 CLIP 模型和 trainer：

```bash
cd /Users/Admin/PyCharmMiscProject

rsync -avP --relative -e "ssh -p 2222" \
  ./model/clip_ad.py \
  ./model/mambaad.py \
  ./model/modules/ \
  ./trainer/clip_ad_trainer.py \
  wmwanghkmu@localhost:/home/wmwanghkmu/ZYH/Mamba/ADer/
```

如果只上传计划 / 交接文档：

```bash
cd /Users/Admin/PyCharmMiscProject

rsync -avP --relative -e "ssh -p 2222" \
  ./plan/20260807-114546-clip-modular-redesign-plan.md \
  ./HANDOFF.md \
  wmwanghkmu@localhost:/home/wmwanghkmu/ZYH/Mamba/ADer/
```

### Stage2C / Supervised 训练命令

生成最省事的 supervised meta：

```bash
cd /home/wmwanghkmu/ZYH/Mamba/ADer
conda activate mamba_ok

python tools/build_supervised_meta.py \
  --root data/mvtec \
  --input meta.json \
  --output meta_supervised.json
```

运行 Stage2A + ARCC：

```bash
cd /home/wmwanghkmu/ZYH/Mamba/ADer
conda activate mamba_ok

CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/clip_ad/clip_ad_mtvecad_stage2a_arcc.py \
  -m train \
  --seed 42
```

运行 Stage2B-3 CSSD/Mamba + text-map ARCC：

```bash
cd /home/wmwanghkmu/ZYH/Mamba/ADer
conda activate mamba_ok

CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/clip_ad/clip_ad_mtvecad_stage2b_cssd_arcc.py \
  -m train \
  --seed 42
```

运行 Full CNN + Mamba + ARCC：

```bash
cd /home/wmwanghkmu/ZYH/Mamba/ADer
conda activate mamba_ok

CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/clip_ad/clip_ad_mtvecad_stage2b_local_global_arcc.py \
  -m train \
  --seed 42
```

运行当前 supervised source-mask Full：

```bash
cd /home/wmwanghkmu/ZYH/Mamba/ADer
conda activate mamba_ok

CUDA_VISIBLE_DEVICES=0 python train.py \
  -c configs/clip_ad/clip_ad_supervised_mask_full.py \
  -m train \
  --seed 42
```

测试已有 checkpoint 通常要单独写 net test config，设置：

```python
self.trainer.resume_dir = 'MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333'
self.model.kwargs['checkpoint_path'] = 'net.pth'
```

## 13. 这个项目里最容易踩的坑

### 坑 1：把 Pixel AUROC 当成定位很好

Pixel AUROC 很容易高，但 Pixel AP/F1 很低，说明热图排序有点用，但真正阈值分割很差。用户经常纠结这个点。要提醒：

```text
Pixel AUROC 高 ≠ lesion mask 精准。
Pixel AP / F1 / top-k hit / AUPRO 更能说明定位质量。
```

### 坑 2：image AUROC 和 pixel 指标脱节

很多实验：

```text
pixel 好，image 差；
image 稍好，pixel 崩。
```

不能只报 Avg，也不能只报一个指标。

### 坑 3：把 AnomalyCLIP 融入主方法

用户多次提到 AnomalyCLIP。结论一直是：

```text
AnomalyCLIP 只能做 external baseline，不要融合进主方法。
```

### 坑 4：继续无脑调 synthetic anomaly

多次尝试 synthetic / wavelet / frequency dual-role 后，结论是：

```text
合成异常容易让模型学 synthetic artifact。
如果没有强证据，不要再把“更像病灶的 synthetic”当唯一主线。
```

### 坑 5：可视化红不等于真实高异常

很多 debug_vis 是单图 min-max 归一化。正常图也可能大片红。必须看：

```text
score distribution
foreground_mask_diagnostic
false_positive_region_diagnostic
localization_point_metrics
```

### 坑 6：foreground mask 对 OCT/retinal 太宽

retinal/OCT 背景有 speckle noise，简单 `8/255` threshold 会把几乎整张图当 foreground。

### 坑 7：不要把所有探索都写成主贡献

当前最适合论文的主线还是：

```text
BiomedCLIP + CNN local evidence + ARCC context calibration
```

频域、FPC、pseudo-memory、region verification 可以作为 analysis / negative study / motivation，不要全堆进方法里。

## 14. 建议下一步

如果用户要继续做实验，最推荐：

```text
先做 Self-Anchored Normality Calibration 的 eval-only 版本。
```

不要先训练。先写工具脚本：

```text
输入：E2 checkpoint + test data
输出：
raw map metrics
self-anchor calibrated map metrics
normal/abnormal score diagnostics
debug_vis_self_anchor
```

如果这个不提升，就不要继续投入太多；可以转向写论文，把 ARCC E2 作为主线，频率分析和失败实验作为动机/讨论。

如果用户要写论文，建议标题/主线不要夸大：

```text
BiomedCLIP-guided Context-Calibrated Local Adapter for Normal-only Medical Anomaly Localization
```

更安全的任务表述：

```text
cross-dataset / cross-modality zero-shot medical anomaly localization under normal-only auxiliary adaptation
```

不要说：

```text
strict unseen-organ zero-shot
```

因为训练数据里可能包含 OASIS brain normal，也加入过 liver/breast/retinal benchmarks 做测试。

## 15. 当前 CLIP supervised mask transfer 分支

更新时间：2026-08-10

当前 CLIP 分支已经从 normal-only 转成 supervised source-mask transfer。
核心代码：

```text
model/clip_ad.py
model/modules/adapters.py
model/mambaad.py
trainer/clip_ad_trainer.py
data/folder_ad.py
configs/clip_ad/clip_ad_supervised_mask_full.py
tools/visualize_clip_ad_heatmaps.py
```

重要设计决定：

```text
当前主线 adapter_type="multilayer_local"。

CLIP 输出 layer 12 / 18 / 24 patch tokens。
layer 12 / 18 输入 CNN local branch，输出 local_delta。
layer 24 是 text-aligned semantic base。
refined_patch = normalize(layer24_patch + alpha * local_delta)。

normal / abnormal text prototypes 只用于 refined_patch 之后的一次 similarity scoring。
先得到 A_raw = sim(refined_patch, abnormal) - sim(refined_patch, normal)。
Mamba/CSSD 在 A_raw 之后做 global response context。
它输入 CLIP layer 24 last_patch，
不接收 text prompt，也不接收 CNN residual 之后的 refined_patch。
ARCC 接收 A_raw、G_mamba 和 refined_patch feature，输出 A_final。
CNN 不直接输出 heatmap。
Mamba/CSSD 不和 CNN 分别输出 heatmap 后再相加。
text prompt 不进入 CNN，也不进入 feature adapter。
ARCC 是唯一直接输出 A_final 的 map-level calibration。

当前主配置默认 image_score_topk_ratio=None，
所以 image_score 使用 A_final 的 max aggregation。
针对 screw 前景整体发红问题，新增 loss_outside_topk：
mask 外最高响应区域用 supervised_outside_topk_weight=0.5 压低。

当前 cross-category 配置使用短目录名：
runs/clipad_YYYYMMDD-HHMMSS
测试输出文件名：
show_test/outputs.npz
```

训练日志里新增 debug 指标：

```text
dbg_refine_cos / dbg_refine_delta_l2:
看 refined patch 是否过度偏离 layer24 patch。

dbg_mamba_context_cos / dbg_mamba_context_delta_l2:
看 ARCC 实际使用的 fused context 与 refined_patch 的差异。

dbg_local_delta_l2 / dbg_local_delta_abs:
看 CNN local branch 是否真的产生局部修正。

dbg_l12_last_cos / dbg_l18_last_cos:
看 layer 12/18 与 layer 24 的语义差距，指导是否只保留 layer18。

dbg_a_raw_mean / dbg_a_raw_max:
看 ARCC 前 text-similarity map 是否有响应。

dbg_a_final_mean / dbg_a_final_max:
看 ARCC 后最终响应。

dbg_arcc_delta_abs / dbg_arcc_delta_ratio / dbg_g_cal_abs / dbg_arcc_lambda:
看 ARCC 修改幅度，判断是否过强或没起作用。

dbg_s_global / dbg_topk_score:
看 image_score 主要来自 global CLIP 还是 final map 的 top-k。

dbg_topk_score_max / dbg_topk_score_top1 / dbg_topk_score_top5:
看 A_final 分别用 max、top 1% mean、top 5% mean 时的局部证据。

dbg_image_score_max / dbg_image_score_top1 / dbg_image_score_top5:
看 S_global + beta * 局部证据后，不同聚合方式的 image score。

dbg_mamba_prior_mean / dbg_mamba_prior_max:
看 last-patch Mamba 全局先验强度。

loss_outside_topk:
看 mask 外 top-k 高响应 suppression 是否起作用。
```

heatmap 可视化诊断：

```bash
python tools/visualize_clip_ad_heatmaps.py \
  --npz runs/<RUN_DIR>/show_test/outputs.npz \
  --data-root data/mvtec \
  --out-dir runs/<RUN_DIR>/show_test/heatmap_vis \
  --count 8
```

输出 PNG 会包含 image / mask / A_raw / A_final / overlay / G_cal / G_mamba / histogram / red-area ratio。
如果 normal 图的 red_ratio_06、red_ratio_08 也很高，说明 final map 可能全图偏红。

sanity check 配置：

```text
configs/clip_ad/clip_ad_supervised_mask_full.py
```

它使用：

```text
data/mvtec/meta_supervised.json
```

注意：

```text
如果 meta_supervised.json 是由 MVTec original test abnormal/mask 加进 train，
再评价同一个 MVTec original test，
这个结果只能说明 pipeline 能学 mask，不能作为论文主结果。
```

正式下一步先跑 MVTec 内部跨类别：

```text
source categories with masks -> held-out target category test
```

新增文件：

```text
tools/build_cross_category_meta.py
configs/clip_ad/clip_ad_supervised_mask_cross_category.py
```

先留出 screw 做 target test：

```bash
cd /home/wmwanghkmu/ZYH/Mamba/ADer
conda activate mamba_ok

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

换目标类别时：

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

跨类别跑通之后，再做跨数据集：

```text
MVTec source masks -> VisA / BTAD / MPDD target test
或 VisA source masks -> MVTec target test
```
