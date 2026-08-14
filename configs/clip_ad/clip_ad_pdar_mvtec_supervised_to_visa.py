import copy

from configs.clip_ad.clip_ad_supervised_mask_pdar_cssd import cfg as base_cfg


class cfg(base_cfg):
    """Train PDAR-CSSD on labeled MVTec source data and evaluate only on VisA."""

    def __init__(self):
        super().__init__()

        # Source domain: labeled MVTec samples used by the current supervised
        # mask-loss experiment. Its internal MVTec test split is never evaluated.
        self.data_train = copy.deepcopy(self.data)
        self.data_train.type = "DefaultAD"
        self.data_train.root = "data/mvtec"
        self.data_train.meta = "meta_supervised.json"
        self.data_train.require_meta = True
        self.data_train.cls_names = []
        self.data_train.train_with_anomaly_masks = True
        # This source-only metadata intentionally adds labeled MVTec anomalies
        # to its train split. The source-side test loader is discarded; the real
        # source-train/VisA-test overlap is checked by CLIPADTrainer.
        self.data_train.enforce_disjoint_train_test = False

        # Target domain: VisA is evaluation-only. No VisA training batch is used.
        self.data_test = copy.deepcopy(self.data)
        self.data_test.type = "DefaultAD"
        # The downloaded VisA layout on the training server has
        # data/split_csv/1cls.csv and data/visa_meta.json at the shared data root.
        self.data_test.root = "data"
        self.data_test.meta = "visa_meta.json"
        self.data_test.require_meta = True
        # VisA contains several extremely small macaroni defects. Bilinear mask
        # resize leaves only sub-0.5 values for 17 samples, which the evaluator
        # would otherwise erase. Preserve those positives and fail loudly if an
        # anomalous test mask is still missing or empty.
        self.data_test.preserve_tiny_masks = True
        self.data_test.require_nonempty_anomaly_mask = True
        self.data_test.cls_names = [
            "pcb1", "pcb2", "pcb3", "pcb4",
            "macaroni1", "macaroni2", "capsules", "candle",
            "cashew", "chewinggum", "fryum", "pipe_fryum",
        ]
        self.data_test.enforce_disjoint_train_test = True

        # BaseTrainer initially needs one data namespace. CLIPADTrainer replaces
        # its loaders with data_train/data_test immediately after initialization.
        self.data = copy.deepcopy(self.data_train)

        # Preserve the exact supervised-loss weights used in the recent PDAR
        # experiment, use max-pixel image-score aggregation, and make the
        # complete PDAR-LSS structure explicit in this cross-domain config.
        mamba_context_kwargs = dict(self.model.kwargs["mamba_context_kwargs"])
        mamba_context_kwargs.update(
            cssd_type="pdar",
            use_selective_scan=True,
            use_cnn_branch=True,
            use_deformable_pool=False,
        )
        self.model.kwargs.update(
            image_score_topk_ratio=None,
            mamba_context_kwargs=mamba_context_kwargs,
            supervised_mask_bce_weight=1.5,
            supervised_mask_dice_weight=2.0,
            supervised_outside_topk_weight=0.1,
        )

        self.trainer.logdir_sub = "pdar_mvtec_supervised_to_visa_max"
        self.trainer.logdir_simple = True
        self.trainer.output_name = "outputs.npz"
