import os

from configs.clip_ad.clip_ad_supervised_mask_full import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()
        self.data.meta = os.environ.get(
            "CLIP_AD_META",
            "meta_cross_category_screw.json",
        )
        self.data.cls_names = []
        self.trainer.logdir_sub = os.environ.get(
            "CLIP_AD_RUN_TAG",
            "clipad",
        )
        self.trainer.logdir_simple = True
        self.trainer.output_name = "outputs.npz"
