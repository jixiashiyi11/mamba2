from argparse import Namespace

import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default):
    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)

        self.fvcore_is = False
        self.seed = 42
        self.size = 336
        self.epoch_full = 1
        self.warmup_epochs = 0
        self.test_start_epoch = 1
        self.test_per_epoch = 1
        self.batch_train = 4
        self.batch_test_per = 4
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.metrics = [
            "mAUROC_sp_max",
            "mAP_sp_max",
            "mF1_max_sp_max",
            "mAUPRO_px",
            "mAUROC_px",
            "mAP_px",
            "mF1_max_px",
        ]

        self.data.type = "DefaultAD"
        self.data.root = "data/mvtec"
        self.data.meta = "meta.json"
        self.data.cls_names = [
            "carpet", "grid", "leather", "tile", "wood",
            "bottle", "cable", "capsule", "hazelnut", "metal_nut",
            "pill", "screw", "toothbrush", "transistor", "zipper",
        ]
        clip_mean = (0.48145466, 0.4578275, 0.40821073)
        clip_std = (0.26862954, 0.26130258, 0.27577711)
        self.data.train_transforms = [
            dict(type="Resize", size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(self.size, self.size)),
            dict(type="ToTensor"),
            dict(type="Normalize", mean=clip_mean, std=clip_std, inplace=True),
        ]
        self.data.test_transforms = self.data.train_transforms
        self.data.target_transforms = [
            dict(type="Resize", size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type="CenterCrop", size=(self.size, self.size)),
            dict(type="ToTensor"),
        ]

        self.model = Namespace()
        self.model.name = "CLIPNormalityAD"
        self.model.kwargs = dict(
            pretrained="openai",
            checkpoint_path="",
            strict=True,
            model_name="ViT-L-14-336",
            img_size=self.size,
            clip_weights_path="",
            require_pretrained=True,
            stage="text_only",
            patch_layer=24,
            surgery_until_layer=None,
            margin=0.2,
            image_score_topk_ratio=None,
            topk_beta=0.5,
            adapter_type="none",
            normal_templates=[
                "a photo of a normal {class_name}",
                "a photo of an intact {class_name}",
                "a photo of a flawless {class_name}",
                "a photo of an undamaged {class_name}",
            ],
            abnormal_templates=[
                "a photo of a damaged {class_name}",
                "a photo of a defective {class_name}",
                "a photo of an anomalous {class_name}",
                "a photo of a broken {class_name}",
            ],
        )

        self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100)
        self.optim.lr = self.lr
        self.optim.kwargs = dict(
            name="adamw",
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
            amsgrad=False,
        )

        self.trainer.name = "CLIPADTrainer"
        self.trainer.logdir_sub = ""
        self.trainer.resume_dir = ""
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(
            name="step", lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42,
            lr_min=self.lr / 1e2, warmup_lr=self.lr / 1e3,
            warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs, cooldown_epochs=0,
            use_iters=True, patience_iters=0, patience_epochs=0, decay_iters=0,
            decay_epochs=int(self.epoch_full * 0.8), cycle_decay=0.1, decay_rate=0.1,
        )
        self.trainer.mixup_kwargs = dict(
            mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0,
            switch_prob=0.5, mode="batch", correct_lam=True, label_smoothing=0.1,
        )
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

        self.loss.loss_terms = []
        self.logging.log_terms_train = [
            dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
            dict(name="data_t", fmt=":>5.3f"),
            dict(name="optim_t", fmt=":>5.3f"),
            dict(name="lr", fmt=":>7.6f"),
            dict(name="total", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_global", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_patch", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_normal_topk", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_consistency", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_image_normal", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_arcc_normal", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_arcc_cal", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_mask_bce", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_mask_dice", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_mask_raw_bce", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_outside_topk", fmt=":>5.3f", add_name="avg"),
            dict(name="loss_image_supervised", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_refine_cos", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_refine_delta_l2", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_mamba_context_cos", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_mamba_context_delta_l2", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_local_delta_l2", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_local_delta_abs", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_l12_last_cos", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_l18_last_cos", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_a_raw_mean", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_a_raw_max", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_a_final_mean", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_a_final_max", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_arcc_delta_abs", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_arcc_delta_ratio", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_arcc_lambda", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_arcc_lambda_learned", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_g_cal_abs", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_mask_bce_normal", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_mask_bce_abnormal", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_mask_dice_positive", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_normal_prob_mean", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_normal_fg_ratio", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_arcc_normal_max_gain", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_raw_mask_gap", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_final_mask_gap", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_batch_normal_count", fmt=":>5.2f", add_name="avg"),
            dict(name="dbg_batch_abnormal_count", fmt=":>5.2f", add_name="avg"),
            dict(name="dbg_batch_positive_mask_count", fmt=":>5.2f", add_name="avg"),
            dict(name="dbg_s_global", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_topk_score", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_topk_score_max", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_topk_score_top1", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_topk_score_top5", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_image_score_max", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_image_score_top1", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_image_score_top5", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_mamba_prior_mean", fmt=":>5.3f", add_name="avg"),
            dict(name="dbg_mamba_prior_max", fmt=":>5.3f", add_name="avg"),
        ]
        self.logging.log_terms_test = [
            dict(name="batch_t", fmt=":>5.3f", add_name="avg"),
        ]
