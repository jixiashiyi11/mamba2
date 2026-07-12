# MambaAD / BiomedCLIP 医疗异常检测代码结构总结

本文档用于把当前仓库的整体逻辑、模块职责、模型结构、训练流程、热力图生成流程和实验配置继承链整理成一份可以直接复制到 Word 或发给 GPT 继续分析的说明。

说明：本仓库基于 ADer/MambaAD 风格实现，包含原始 MambaAD 工业/通用异常检测模型，也包含你后来加入的医疗场景 BiomedCLIP、zero-shot、local adapter、dual branch、TGLRA、CNN global auxiliary 等实验分支。根据当前对话上下文，重点实验线是 `configs/mambaad/mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_mixed_wavelet_mild_e15.py` 及其继承链。


## 1. 仓库整体定位

这个仓库是一个无监督/弱监督异常检测框架。原始 MambaAD 的核心思想是：

1. 用冻结的预训练 CNN/ResNet 作为 teacher encoder，提取正常图像的多尺度特征。
2. 用 Mamba decoder / LSS 模块作为 student 或重建分支，学习重建正常样本的 teacher features。
3. 测试时，如果某个区域异常，student 对该区域重建得不好，teacher-student feature 差异变大，于是得到 anomaly map。

你现在的医疗实验线做了很大改造：

1. 用冻结的 BiomedCLIP 作为医学图像-文本特征提取器。
2. 从 BiomedCLIP 中取整图 image feature、patch tokens、normal/abnormal text prompt features。
3. 只训练一个较轻的 local adapter / CSSD / CNN decoder / relation branch，让它在正常图上不要乱亮，在合成局部异常上学会定位。
4. 当前 local adapter 分支的 heatmap 不再是传统 teacher-student feature difference，而是 adapter 或 head 直接输出的 anomaly logits / anomaly map。


## 2. 入口和运行流程

主入口是 `run.py`。

整体执行顺序：

1. `argparse` 读取命令行参数：
   - `-c / --cfg_path`：配置文件路径。
   - `-m / --mode`：`train` 或 `test`。
   - `opts`：命令行覆盖配置，例如 `data.cls_names=brain`。
2. `get_cfg(cfg_terminal)`：
   - 动态 import 配置文件。
   - 实例化配置类 `cfg()`。
   - 把命令行参数写入 cfg。
   - 解析 `opts`，支持 `path.key=value` 形式覆盖嵌套配置。
3. `run_pre(cfg)`：
   - 可选等待 GPU 显存或 sleep。
4. `init_training(cfg)`：
   - 初始化 CUDA、随机种子、DDP、batch size。
5. `init_checkpoint(cfg)`：
   - 创建 logdir。
   - 处理 resume/test checkpoint。
   - 创建 logger 和 TensorBoard writer。
6. `get_trainer(cfg)`：
   - 根据 `cfg.trainer.name` 从 trainer registry 取训练器类。
7. `trainer.run()`：
   - 如果 mode 是 train，进入训练循环。
   - 如果 mode 是 test，直接测试。

辅助脚本：

1. `run.sh`：DDP 启动模板，调用 `torch.distributed.launch run.py -c $1 -m $2`。
2. `runs_single_class.py`：按单类别循环跑多个数据集类别，例如 MVTec/VisA 单类实验。
3. `script.py`：PyCharm 示例脚本，不参与主逻辑。


## 3. Registry 注册机制

核心文件：`util/registry.py`

仓库使用简单 registry 模式：

1. `MODEL = Registry('Model')`
2. `TRAINER = Registry('Trainer')`
3. `LOSS = Registry('Loss')`

每个模块通过装饰器注册，例如：

1. `@MODEL.register_module`
2. `@TRAINER.register_module`
3. `@LOSS.register_module`

配置里只需要写字符串名，例如：

1. `self.model.name = 'mambaad_biomedclip_local_adapter'`
2. `self.trainer.name = 'MAMBAADZeroShotTrainer'`
3. `loss_terms = [dict(type='CosLoss', name='pixel')]`

运行时通过 `get_model(cfg.model)`、`get_trainer(cfg)`、`get_loss_terms(...)` 动态实例化。


## 4. 配置系统

配置入口：`configs/__init__.py`

配置文件本质是 Python class，通常继承多个 base config。所有配置最终都会变成一个 `Namespace` 树。

基础配置：

1. `configs/__base__/cfg_common.py`
   - 定义通用训练参数、评估指标、optimizer、scheduler、logging、debug_eval、synthetic_local_anomaly 等默认字段。
   - 默认 trainer 是 `ViTADTrainer`，但具体模型配置会覆盖。

2. `configs/__base__/cfg_dataset_default.py`
   - 定义默认数据集类型、路径、类别列表、transform。
   - 支持 MVTec、VisA、MVTec3D、medical、RealIAD 等类别名。

3. `configs/__base__/cfg_model_mambaad.py`
   - 原始 MambaAD 默认 teacher/student/model 配置：
     - `model_t = timm_wide_resnet50_2`
     - `model_s = de_wide_resnet50_2`
     - `model.name = 'mambaad'`


## 5. 你当前重点实验配置继承链

当前对话重点配置：

`configs/mambaad/mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_mixed_wavelet_mild_e15.py`

继承链：

1. `mambaad_medical.py`
2. `mambaad_medical_aux_train_balanced.py`
3. `mambaad_medical_aux_train_balanced_loss_ablation.py`
4. `mambaad_medical_aux_train_balanced_loss_B_cons_0p1.py`
5. `mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_e25.py`
6. `mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_wavelet_e15.py`
7. `mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_mixed_wavelet_e15.py`
8. `mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_mixed_wavelet_mild_e15.py`

关键覆盖关系：

1. `mambaad_medical.py`
   - 数据：`data/medical`
   - 类别：brain、liver、retinal。
   - 模型：`mambaad_zsad`
   - trainer：`MAMBAADZeroShotTrainer`
   - 使用 BiomedCLIP prompt：
     - `prompt_normal`
     - `prompt_abnormal`
     - `class_prompts`

2. `mambaad_medical_aux_train_balanced.py`
   - 把训练集换成 normal-only 辅助训练集：
     - `data.type = 'FolderNormalAD'`
     - `data.root = 'data/medical_aux_train_balanced'`
     - `data.cls_names = ['good']`
   - 同时保留 `data_test = data/medical` 做测试。
   - 加入 `good` 类 prompt。

3. `loss_ablation.py`
   - 设置 `adaptive_mc_kwargs`：
     - normal alignment 权重。
     - margin 权重。
     - token consistency 权重。
     - score separation 权重。

4. `B_cons_0p1.py`
   - 把 `lambda_cons` 改成 `0.1`。

5. `local_adapter_e25.py`
   - 这是重要转折：模型从 `mambaad_zsad` 改成 `mambaad_biomedclip_local_adapter`。
   - 不再使用 ResNet visual encoder + AdaptiveMCLoss 主线，而是使用 BiomedCLIP patch tokens + local adapter。
   - 开启 synthetic local anomaly。
   - 开启 debug_eval。

6. `wavelet_e15.py`
   - epoch 从 25 改到 15。
   - 合成异常从 ellipse 改成 wavelet。

7. `mixed_wavelet_e15.py`
   - 合成异常模式改成 `mixed_wavelet`。
   - `wavelet_mix_prob = 0.5`。

8. `mild_e15.py`
   - 进一步把合成异常调弱：
     - `wavelet_mix_prob = 0.25`
     - `min_area = 0.001`
     - `max_area = 0.035`
     - `noise_std = 0.08`
     - `intensity_delta = 0.15`
     - wavelet 扰动也变小。
   - `normal_topk_loss_weight` 提高到 `0.2`，更强惩罚正常图上的高异常响应。

如果用这条配置启动，实际模型和 trainer 是：

1. `cfg.trainer.name = 'MAMBAADZeroShotTrainer'`
2. `cfg.model.name = 'mambaad_biomedclip_local_adapter'`
3. 模型类是 `MAMBAADBiomedCLIPLocalAdapter`
4. 训练器类是 `MAMBAADZeroShotTrainer`


## 6. 数据加载模块

核心文件：`data/folder_ad.py`

提供两类 dataset：

1. `FolderNormalADDataset`
   - 用于 normal-only 训练集。
   - 预期结构类似：
     - `root/train/good/*.png`
   - 每张图的 mask 都是全 0。
   - 返回字段：
     - `img`
     - `img_mask`
     - `cls_name`
     - `anomaly`
     - `img_path`
     - `mask_path`

2. `MetaADDataset`
   - 用于 ADer 风格 `meta.json` 数据集。
   - 支持 `train` 和 `test` split。
   - 如果存在 mask path，则读取真实 mask；否则返回全 0 mask。
   - 也兼容 MVTec 风格目录扫描。

`get_loader(cfg)`：

1. 根据 `cfg.data.type` 创建 train/test dataset。
2. 根据 DDP 情况创建 sampler。
3. 返回 train_loader 和 test_loader。
4. 设置 `cfg.trainer.iter_full = epoch_full * len(train_loader)`。

医疗辅助训练场景中，`MAMBAADZeroShotTrainer._rebuild_cross_domain_loaders()` 会特殊处理：

1. 用 `cfg.data_train` 加载 source normal-only train loader。
2. 用 `cfg.data_test` 加载 target medical test loader。
3. 训练类通常是 `good`，测试类是 `brain/liver/retinal`。


## 7. BaseTrainer 通用训练骨架

核心文件：`trainer/_base_trainer.py`

`BaseTrainer` 负责所有模型共用流程：

1. 构建模型：
   - `self.net = get_model(cfg.model)`
   - 放到 CUDA。
   - 可选加载 checkpoint。

2. 构建 optimizer：
   - `get_optim(cfg.optim.kwargs, self.net, lr=cfg.optim.lr)`

3. 构建 loss：
   - `get_loss_terms(cfg.loss.loss_terms)`

4. 构建 DDP：
   - 支持 Native DDP / Apex DDP。

5. 构建数据：
   - `self.train_loader, self.test_loader = get_loader(cfg)`

6. 构建 scheduler：
   - `get_scheduler(cfg, self.optim)`

7. 构建 evaluator：
   - `get_evaluator(cfg.evaluator)`

8. 训练循环：
   - 每个 iteration：
     - scheduler step。
     - 取 batch。
     - `set_input()`。
     - `optimize_parameters()`。
     - 写日志。
   - 每个 epoch 结束：
     - 根据配置决定 test 或 test_ghost。
     - 保存 checkpoint。

9. 保存 checkpoint：
   - `ckpt.pth`：包含 net、optimizer、scheduler、scaler、iter、epoch、metric_recorder。
   - `net.pth`：只保存网络权重。
   - `net_{epoch}.pth`：每隔 test_per_epoch 保存一次。

子类必须实现：

1. `set_input`
2. `forward`
3. `optimize_parameters`
4. `test`


## 8. MAMBAADTrainer：原始 MambaAD 训练器

核心文件：`trainer/mambaad_trainer.py`

`MAMBAADTrainer` 对应较原始的 teacher/student MambaAD。

训练逻辑：

1. `set_input()`：
   - 读取 `img`、`img_mask`、`cls_name`、`anomaly`。

2. `forward()`：
   - 调用：
     - `self.feats_t, self.feats_s, self.f_global = self.net(self.imgs, self.cls_name, return_teacher_features=True)`
   - teacher features 是 frozen encoder 输出。
   - student features 是 Mamba decoder 重建输出。

3. `optimize_parameters()`：
   - `loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)`
   - 可选 adaptive_mc：
     - 提前用 BiomedCLIP text encoder 缓存 normal/abnormal text prior。
     - 用 `f_global` 对齐 normal text prior。
   - 可选 L1 penalty。
   - total loss = pixel loss + L1 + adaptive_mc。

4. `test()`：
   - 计算 teacher/student feature 差异。
   - 用 `Evaluator.cal_anomaly_map()` 生成 anomaly map。
   - 聚合 results 后算指标。

注意：如果当前配置是 local adapter，这个 trainer 不走；当前 local adapter 走的是 `MAMBAADZeroShotTrainer`。


## 9. MAMBAADZeroShotTrainer：当前医疗实验主训练器

核心文件：`trainer/mambaad_trainer.py`

这个 trainer 名字叫 ZeroShot，但它同时服务于：

1. `mambaad_zsad`
2. `mambaad_biomedclip_local_adapter`
3. `mambaad_biomedclip_dual_branch_adapter`
4. TGLRA 和 CNN global aux 变体

核心输入：

1. `self.imgs`
2. `self.imgs_mask`
3. `self.cls_name`
4. `self.anomaly`
5. `self.img_path`
6. `self.mask_path`

训练时 `forward()` 的逻辑：

1. 根据 `_get_model_cls_names()` 得到：
   - `score_cls_names`：用于 anomaly scoring 的类别名。
   - `adapter_cls_names`：用于 adapter conditioning 的类别名。
2. 如果模型处于 training：
   - `self.loss_dict = self.net(self.imgs, cls_names=score_cls_names, adapter_cls_names=adapter_cls_names)`
   - 模型返回一个 loss dict，里面必须有 `total`。
3. 如果开启 `synthetic_local_anomaly.enabled`：
   - `_make_synthetic_local_anomaly_batch()` 生成合成异常图和 synthetic mask。
   - 再调用一次模型：
     - `return_anomaly_map=True`
     - `compute_label_free=False`
   - 得到 synthetic anomaly map。
   - `_synthetic_local_anomaly_loss()` 对 synthetic map 和 synthetic mask 做 BCE + Dice + 可选 compactness 约束。
   - 把 synthetic loss 加到 `loss_dict['total']`。
4. `optimize_parameters()` 对 `self.total_loss` 反向传播。

测试时 `test()` 的逻辑：

1. `self.forward()` 得到：
   - `self.anomaly_map`
   - `self.image_score`
2. 收集：
   - `imgs_masks`
   - `anomaly_maps`
   - `image_scores`
   - `cls_names`
   - `anomalys`
   - adapter debug 信息
   - img/mask path
   - foreground mask
3. 如果 debug_eval 开启：
   - `DebugEvalHelper.add_vis_batch()` 收集可视化样本。
   - 测试结束后输出 debug csv 和 debug_vis 图片。
4. `_emit_metric_summary_table()` 输出指标表。


## 10. 合成局部异常逻辑

核心函数：`MAMBAADZeroShotTrainer._make_synthetic_local_anomaly_batch()`

输入是正常图 `self.imgs`，输出：

1. `synth_imgs`：加入伪异常后的图。
2. `masks`：伪异常区域 mask。

步骤：

1. 反归一化图像到 0-1：
   - `imgs_01 = imgs * std + mean`
2. 根据强度阈值生成 foreground mask：
   - 避免在纯黑背景上造异常。
3. 在 foreground 内随机采样中心点。
4. 根据 lesion mode 生成 mask：
   - `ellipse`：椭圆硬 mask。
   - `soft_brain`：多个高斯 soft blob，更像脑部软病灶。
   - 其它 mode 默认走 ellipse/spatial 逻辑。
5. 根据 lesion mode 修改图像：
   - `wavelet`：Haar DWT 后扰动 LL/edge/texture 系数，再 IDWT 回图像。
   - `mixed_wavelet`：随机在 spatial lesion 和 wavelet lesion 之间选择。
   - 其它：`_apply_spatial_synthetic_lesion()`，即局部亮度/噪声扰动。

合成异常 loss：`_synthetic_local_anomaly_loss()`

1. `logits = anomaly_map / score_temperature`
2. `loss_bce = BCEWithLogits(logits, synth_mask)`
3. `loss_dice = Dice loss`
4. 可选：
   - `outside_suppression_weight`：抑制 synthetic mask 外 foreground 区域的异常概率。
   - `area_sparsity_weight`：限制预测区域不要比目标 mask 大太多。
5. 总 synthetic loss：
   - `bce_weight * BCE + dice_weight * Dice + outside_weight * outside + area_weight * area`

重要注意：

部分配置中写了 `lesion_mode = 'multi_weak'`，并设置 blur/contrast/copypaste 等参数；但当前 trainer 代码只显式处理 `wavelet` 和 `mixed_wavelet`，否则走 spatial lesion。因此如果运行 `multi_weak` 配置，现有代码不会真正执行 blur/contrast/copypaste 多类型扰动，除非后续补实现。


## 11. 原始 MambaAD 模型结构

核心类：`MAMBAAD`

主要模块：

1. `net_t`
   - teacher encoder，通过 `get_model(model_t)` 构建。
   - 通常是 timm ResNet。
   - 训练时冻结。

2. `MFF_OCE`
   - 多尺度 feature fusion 模块。
   - 把 teacher 多层 feature 融合成 bottleneck feature。

3. `MambaUPNet`
   - Mamba decoder。
   - 由多层 `LSSLayer_up` 组成。
   - 输出重建特征列表。

4. `FrozenBiomedTextEncoder`
   - 用 BiomedCLIP text encoder 生成 class prompt embedding。
   - 用作 AdaLN semantic condition。

5. `text_proj`
   - 把 bottleneck 全局池化特征投影到 512 维，用于 normal alignment。

前向流程：

1. `feats_t = net_t(imgs)`。
2. `fused_feats = MFF_OCE(feats_t)`。
3. `f_global = text_proj(GAP(fused_feats))`。
4. 如果传入 `cls_names`，用 frozen BiomedCLIP text encoder 编码 class prompts，得到 `c_embed`。
5. `reconstructed_features = net_s(fused_feats, c_embed)`。
6. 如果 `return_teacher_features=True`，返回 `(feats_t, reconstructed_features, f_global)`。


## 12. Mamba / LSS / CSSD 底层模块

这些模块都在 `model/mambaad.py`。

1. `HSCANS`
   - 负责把二维 feature map 展平成序列时的扫描顺序。
   - 支持：
     - sweep
     - scan
     - zorder
     - zigzag
     - hilbert
   - 同时保存正向 index 和 inverse index。

2. `SS2D`
   - 2D selective scan 模块。
   - 把输入按多个方向扫描。
   - 使用 `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`。
   - 支持 `num_direction`，例如 4 或 8。

3. `HSSBlock`
   - 一个 Hybrid State Space block。
   - 包含 LayerNorm、SS2D、DropPath。
   - 支持 AdaLN modulation：
     - 输入条件 `c` 是 512 维语义向量。
     - 生成 gamma/beta 调制归一化后的 token。

4. `DeformableAttnRes`
   - 用 deformable convolution 做局部动态残差。
   - 输入 query feature，预测 offset/mask，再对 pool feature 做 deform conv。

5. `LSSModule`
   - Locality-Enhanced State Space 模块。
   - 结构：
     - 多个 HSSBlock 捕捉长程依赖。
     - DeformableAttnRes 捕捉局部可变形关系。
     - 残差连接。

6. `LSSLayer_up`
   - 多个 LSSModule 组成一层 decoder。
   - 可选 upsample。

7. `MambaUPNet`
   - 多尺度 Mamba decoder。
   - 原始 MambaAD 中用于 student 重建。

8. `CSSD`
   - 当前 BiomedCLIP local adapter 的核心 token refinement 模块。
   - 输入：
     - `v_raw`: `(B, L, D)` patch tokens。
     - `semantic_embedding`: 可选语义条件。
     - `spatial_shape`: patch grid。
   - 内部把 tokens reshape 成 `(B, H, W, D)`，经过多个 LSSModule，然后输出 refined tokens。


## 13. BiomedCLIP 编码模块

1. `FrozenBiomedTextEncoder`
   - 用 `open_clip.create_model_and_transforms(model_name)` 加载 BiomedCLIP。
   - 只使用 text encoder。
   - 支持：
     - normal prompt
     - abnormal prompt
     - class prompt
   - 输出：
     - `t_norm`
     - `t_abn`
     - class semantic embedding。
   - 全部冻结。

2. `FrozenBiomedCLIPPatchEncoder`
   - 用 BiomedCLIP 同时提取：
     - 整图 `image_features`
     - patch-level `tokens`
     - patch grid shape。
   - `_prepare_images()` 会把 ImageNet normalization 的输入反归一化，再按 BiomedCLIP mean/std 重新归一化，并 resize 到 BiomedCLIP 期望尺寸。
   - `_tokens_from_feature_tensor()` 兼容不同 open_clip visual trunk 输出格式。
   - `_apply_projection()` 尝试把 visual patch tokens 投影到 text embedding 维度。
   - `encode_text_pairs()` 返回 normal/abnormal prompt embeddings。
   - `encode_image_and_patches()` 返回 `(image_features, tokens, grid_shape)`。


## 14. 当前主线模型：MAMBAADBiomedCLIPLocalAdapter

当前重点配置实际使用这个类。

模型组成：

1. `biomedclip`
   - `FrozenBiomedCLIPPatchEncoder`
   - 完全冻结。

2. `local_adapter`
   - `CSSD`
   - 可训练。
   - 用 LSS/Mamba 风格模块 refine BiomedCLIP patch tokens。

3. `local_head`
   - `nn.Linear(visual_dim, 1)`
   - 对每个 refined patch token 输出一个 patch logit。
   - 可训练。

训练参数：

1. 冻结 BiomedCLIP。
2. 训练 CSSD local_adapter。
3. 训练 local_head。

forward 流程：

1. BiomedCLIP 提取：
   - `image_features`
   - `tokens`
   - `spatial_shape`
   - `t_norm`
   - `t_abn`
2. 计算 image_score：
   - `sim(image, abnormal) - sim(image, normal)`
   - 这个 score 主要用于 image-level 诊断。
3. anomaly map：
   - 如果 `eval_adapter_mode == 'bypass'`：
     - 不走 adapter。
     - 直接用 `sim_normal - sim_abnormal` 作为 patch logits。
   - 否则：
     - `refined = local_adapter(tokens)`
     - `patch_logits = local_head(refined)`
   - patch logits reshape 成 grid，再 bilinear upsample 到输入图像尺寸。
4. 训练时：
   - 如果 `compute_label_free=True`，用 `_localization_losses()` 在正常图上抑制异常响应。
   - 如果 `compute_label_free=False`，只返回 anomaly_map/image_score，供 synthetic loss 使用。
5. 测试时：
   - 返回 `(anomaly_map, image_score)`。

正常图 localization loss：

1. `_foreground_masks()` 根据图像强度生成 foreground/background/edge/interior。
2. `normal_topk`：foreground 内 top-k 高响应 softplus，惩罚正常图上最亮的异常点。
3. `background_loss`：背景区域响应不应过高。
4. `edge_loss`：前景边缘响应不应过高。
5. total：
   - `normal_topk_weight * normal_topk + background_weight * background + edge_weight * edge`

这条线的直观含义：

在正常训练图上，模型被要求整张图不要亮；在 synthetic 异常图上，模型被要求 synthetic mask 区域亮、其它区域不亮。


## 15. 其它模型变体

1. `MAMBAADZeroShot`
   - 使用 frozen visual encoder + frozen BiomedCLIP text encoder。
   - 用 CSSD refine visual tokens。
   - 用 AdaptiveMCLoss 做 label-free normal alignment、margin、token consistency。
   - anomaly map 来自 refined visual tokens 与 normal/abnormal text embeddings 的相似度差。

2. `MAMBAADBiomedCLIPDualBranchAdapter`
   - 在 local adapter 基础上加入 dual branch：
     - CNN local decoder：直接从 patch tokens 解码局部 anomaly map。
     - CSSD image branch：用 CSSD refined tokens 计算 image-level anomaly score。
     - learnable text delta：对 normal/abnormal text prototypes 做小幅可学习修正。
     - semantic gate：用 text similarity map 对 CNN logits 做门控增强。
   - 训练包含：
     - 正常图 localization loss。
     - CSSD image branch 正常图 BCE。
     - text prototype regularization。

3. `MAMBAADBiomedCLIPTGLRANoMamba`
   - TGLRA 表示 Text-Guided Local Relation Adapter。
   - 不使用 CSSD/Mamba image branch。
   - 用 `TextGuidedLocalRelationBranch` 计算 token 与邻域 token、semantic score、semantic diff 的局部关系。
   - 主要训练 relation branch。

4. `MAMBAADBiomedCLIPTGLRAFull`
   - 结合 CSSD global tokens、TGLRA relation tokens、原始 tokens 和 semantic patch score。
   - 用 fusion MLP + fusion head 输出 anomaly map。
   - 属于更完整的 global-local fusion 版本。

5. `MAMBAADBiomedCLIPCNNGlobalAuxAdapter`
   - 基于 dual branch。
   - 增加 global auxiliary gate：
     - CSSD branch 产生 global patch scores。
     - global gate 与 semantic gate 一起调制 CNN logits。


## 16. AdaptiveMCLoss

核心文件：`loss/adaptive_mc_loss.py`

用于 label-free normal-only 训练，主要服务 `MAMBAADZeroShot` 这条线。

输入：

1. `v_refined`
2. `v_raw`
3. `t_norm`
4. `t_abn`
5. 可选 `f_global`

内部 loss：

1. `loss_normal_align`
   - 希望全局视觉特征靠近 normal text embedding。
   - `1 - sim(f_global, t_norm)`

2. `loss_margin`
   - 希望 normal similarity 比 abnormal similarity 高至少 adaptive margin。
   - margin 会根据 normal/abnormal 相似度混淆程度动态变化。

3. `loss_token_consistency`
   - 希望 refined tokens 不要偏离 raw frozen visual tokens 太多。

4. `loss_score_separation`
   - 可选，用 top-k token anomaly scores 抑制正常图上的高异常响应。

总 loss：

`lambda_normal_align * normal_align + lambda_margin * margin + lambda_cons * token_consistency + lambda_score_separation * score_separation`


## 17. 普通 loss 模块

核心目录：`loss/`

1. `base_loss.py`
   - `L1Loss`
   - `L2Loss`
   - `CosLoss`
   - `KLLoss`
   - `FocalLoss`
   - `SSIMLoss`
   - `FFTLoss`
   - `SegmentCELoss`
   - 等基础 loss。

2. `cls_loss.py`
   - `CE`
   - `LabelSmoothingCE`
   - `SoftTargetCE`
   - `CLSKDLoss`

3. `gan_loss.py`
   - GAN 相关 loss。

4. `adaptive_mc_loss.py`
   - 你医疗 zero-shot 语义约束实验的核心 loss。

当前 local adapter 主线并不直接用 `AdaptiveMCLoss`，而是在模型内部 `_localization_losses()` + trainer 内 synthetic loss。


## 18. 评估指标模块

核心文件：`util/metric.py`

`Evaluator` 负责 image-level 和 pixel-level 指标。

主要指标：

1. image-level：
   - `mAUROC_sp_max`
   - `mAP_sp_max`
   - `mF1_max_sp_max`

2. pixel-level：
   - `mAUROC_px`
   - `mAP_px`
   - `mF1_max_px`
   - `mAUPRO_px`
   - `mIoU_max_px`
   - 以及阈值范围内的 F1/Acc/IoU。

`cal_anomaly_map()`：

1. 原始 teacher/student 模型使用。
2. 对每层 feature：
   - cosine distance 或 L2 distance。
   - 上采样到输出尺寸。
   - 多层 add 或 multiply 聚合。
   - 可选 Gaussian smoothing。

当前 local adapter 模型不使用 `cal_anomaly_map()` 生成图，因为模型直接返回 anomaly map。


## 19. DebugEval 和热力图可视化

核心文件：`util/debug_eval.py`

`DebugEvalHelper` 用于更深入诊断：

1. 保存 debug records CSV。
2. 保存 score distribution CSV。
3. 保存 foreground diagnostic CSV。
4. 保存 false positive region diagnostic CSV。
5. 保存 debug visualization panel。

测试阶段调用位置：

1. `MAMBAADZeroShotTrainer.test()`
2. 每个 batch 调用：
   - `debug_helper.add_vis_batch(...)`
3. 测试结束：
   - `debug_helper.save_visualizations()`
   - `debug_helper.write_and_summarize(...)`

你截图里的五联图来自 `_save_sample_visualization()`。

从左到右：

1. 原图：
   - 输入 tensor 反归一化。
2. GT mask：
   - 白色是真实异常区域。
3. anomaly heatmap：
   - 对模型输出 `anomaly_map` 做单图 min-max normalization。
   - 用 `matplotlib.cm.jet` 上色。
4. heatmap overlay：
   - `0.55 * img + 0.45 * heat`
5. GT boundary overlay：
   - 在原图上用红色画真实 mask 边界。

如果 foreground debug 开启且样本中有 foreground mask，后面还会追加：

1. foreground mask
2. foreground-masked heatmap
3. foreground-masked overlay
4. percentile heatmap
5. percentile overlay
6. eroded foreground mask
7. eroded foreground heatmap
8. eroded foreground overlay


## 20. Optimizer 和 Scheduler

核心文件：

1. `optim/__init__.py`
2. `optim/scheduler.py`

Optimizer：

1. 支持 SGD、Adam、AdamW、RAdam、NAdam、AdamP、SGDP、Adafactor、Adahessian 等。
2. 如果有 weight_decay，默认会把 bias 和 BN 参数放进 no_decay group。
3. 支持 lookahead，例如 optimizer 名称中带 lookahead 前缀。

Scheduler：

1. 支持：
   - cosine
   - tanh
   - step
   - plateau
2. 支持按 iter 或 epoch 计算 warmup/decay。
3. 当前医疗配置多使用 step scheduler。


## 21. 当前主线的完整训练数据流

以 `mild_e15` local adapter 配置为例：

1. DataLoader 读取正常训练图：
   - `img`: ImageNet normalized tensor。
   - `img_mask`: 全 0。
   - `cls_name`: 通常是 `good`。

2. Trainer 调用模型：
   - `self.net(self.imgs, cls_names=score_cls_names, adapter_cls_names=adapter_cls_names)`

3. 模型内部：
   - 反归一化到 0-1。
   - 重归一化为 BiomedCLIP 输入。
   - BiomedCLIP frozen encoder 输出 patch tokens。
   - CSSD local adapter refine tokens。
   - local_head 输出 patch logits。
   - logits reshape + upsample 得到 anomaly_map。

4. 正常图 loss：
   - foreground 内 top-k 高响应被压低。
   - background 被压低。
   - foreground edge 被压低。

5. Trainer 生成 synthetic anomaly：
   - 从正常图中生成局部 synthetic lesion。
   - 生成 synthetic mask。

6. 模型再次前向 synthetic 图：
   - 返回 synthetic anomaly_map。

7. synthetic loss：
   - synthetic mask 内希望亮。
   - synthetic mask 外希望不亮。
   - BCE + Dice + 可选 outside suppression + area sparsity。

8. 总 loss：
   - 正常图 localization loss + synthetic local anomaly loss + 可选 auxiliary losses。

9. 反向传播：
   - BiomedCLIP 冻结。
   - 主要更新 local_adapter 和 local_head。


## 22. 当前主线的完整测试数据流

1. DataLoader 读取 medical test 图：
   - 类别可能是 brain/liver/retinal。
   - 有真实 mask。

2. Trainer 调用模型：
   - `self.anomaly_map, self.image_score = self.net(...)`

3. 模型输出：
   - `anomaly_map`: pixel-level heatmap。
   - `image_score`: image-level anomaly score。

4. Trainer 收集所有 batch。

5. Evaluator 计算：
   - image-level AUROC/AP/F1。
   - pixel-level AUROC/AP/F1/AUPRO/IoU。

6. DebugEval 保存：
   - 可视化 panel。
   - CSV 诊断表。
   - foreground/false-positive 分析。


## 23. 当前模型 heatmap 的含义

如果模型是 `mambaad_biomedclip_local_adapter`：

1. heatmap 数值本质是 `local_head(refined_patch_tokens)` 的输出。
2. refined patch tokens 来自：
   - frozen BiomedCLIP patch tokens
   - 经过 CSSD/LSS/Mamba-style adapter。
3. heatmap 不是概率，训练时会作为 logits 使用。
4. 可视化时会做归一化，所以颜色只代表该图内部相对高低，不一定代表跨图绝对分数。
5. 图像级 score 和像素级 map 不是完全同一个来源：
   - image_score 来自 BiomedCLIP image feature 与 abnormal/normal text feature 的差异。
   - anomaly_map 来自 local adapter/local head。


## 24. 容易混淆的点

1. `MAMBAADTrainer` 和 `MAMBAADZeroShotTrainer` 不一样。
   - 原始 MambaAD 用 `MAMBAADTrainer`。
   - 当前医疗 local adapter 用 `MAMBAADZeroShotTrainer`。

2. `mambaad_zsad` 和 `mambaad_biomedclip_local_adapter` 不一样。
   - `mambaad_zsad` 用 AdaptiveMCLoss 和 text similarity map。
   - `mambaad_biomedclip_local_adapter` 直接训练 local head 输出 heatmap。

3. 当前 local adapter heatmap 不是 teacher/student 差异图。

4. Debug panel 第 5 张图是 GT boundary，不是预测边界。

5. `normal_minus_abnormal` 与 `abnormal_minus_normal` 方向要特别注意。
   - 有些分支用 `sim_normal - sim_abnormal`。
   - 有些 image score 用 `sim_abnormal - sim_normal`。
   - DebugEval 里也会计算 old/reverse 两个方向的 score sweep。

6. `multi_weak` 配置当前未在 trainer 中显式实现。

7. 可视化 heatmap 是 min-max 或 percentile 归一化后的颜色，不等于原始 logits 的绝对尺度。


## 25. 一句话概括当前代码

这个仓库最初是一个 MambaAD teacher-student 特征重建式异常检测框架；你当前的医疗实验线把它改造成了一个冻结 BiomedCLIP 的医学视觉-文本异常定位框架：BiomedCLIP 负责提供医学语义和 patch tokens，CSSD/Mamba-style local adapter 或后续 dual-branch/TGLRA 模块负责学习局部异常响应，训练时用 normal-only 抑制损失和 synthetic local anomaly 定位损失共同塑造 heatmap，测试时直接输出 anomaly map、image score，并通过 DebugEval 生成热力图、指标和 foreground/false-positive 诊断。

