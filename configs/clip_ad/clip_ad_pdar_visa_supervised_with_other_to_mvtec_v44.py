"""V44: train V43 on all labeled VisA images and test on MVTec AD."""

import copy

from configs.clip_ad.clip_ad_pdar_mvtec_supervised_to_visa_v43_mamba_candidate_hard_negative import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """Reverse the V43 cross-domain direction without changing the model."""

    def __init__(self):
        super().__init__()

        # Source: all labeled VisA images.  The generated metadata combines the
        # official normal-only train split with the complete labeled test split,
        # including the 35 images currently stored under data/visa_other.
        self.data_train = copy.deepcopy(self.data_train)
        self.data_train.root = "data"
        self.data_train.meta = "visa_meta_supervised_v44.json"
        self.data_train.require_meta = True
        self.data_train.cls_names = []
        self.data_train.train_with_anomaly_masks = True
        self.data_train.preserve_tiny_masks = True
        self.data_train.require_nonempty_anomaly_mask = True
        self.data_train.enforce_disjoint_train_test = False

        # Target: the untouched official MVTec AD test split.  Its train split
        # is present only for the metadata disjointness check and is never used.
        self.data_test = copy.deepcopy(self.data_test)
        self.data_test.root = "data/mvtec"
        self.data_test.meta = "meta.json"
        self.data_test.require_meta = True
        self.data_test.cls_names = [
            "bottle",
            "cable",
            "capsule",
            "carpet",
            "grid",
            "hazelnut",
            "leather",
            "metal_nut",
            "pill",
            "screw",
            "tile",
            "toothbrush",
            "transistor",
            "wood",
            "zipper",
        ]
        self.data_test.preserve_tiny_masks = False
        self.data_test.require_nonempty_anomaly_mask = True
        self.data_test.enforce_disjoint_train_test = True

        # BaseTrainer reads self.data during initialization; keep it aligned
        # with the source training namespace used by CLIPADTrainer.
        self.data = copy.deepcopy(self.data_train)

        # Pin the restored baseline explicitly.  Later experiments may add
        # optional hard-normal prototype arguments to the shared model, but a
        # V44 run must always keep that V46-only branch disabled.
        self.model.kwargs.update(
            mamba_hard_normal_prototype_enabled=False,
            mamba_hard_normal_ce_weight=0.0,
            mamba_hard_normal_rank_weight=0.0,
        )

        self.trainer.resume_dir = ""
        self.model.kwargs["checkpoint_path"] = ""
        self.trainer.logdir_sub = "pdar_visa_supervised_with_other_to_mvtec_v44"
        self.trainer.logdir_simple = True
        self.trainer.output_name = "outputs.npz"
