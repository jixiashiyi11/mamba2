from configs.clip_ad.clip_ad_mtvecad import cfg as stage1_cfg


class cfg(stage1_cfg):
    def __init__(self):
        super().__init__()
        self.model.kwargs.update(
            stage="multi_level_text",
            levels=(6, 12, 18, 24),
            multi_level_fusion="mean",
            layer_weights=None,
        )
