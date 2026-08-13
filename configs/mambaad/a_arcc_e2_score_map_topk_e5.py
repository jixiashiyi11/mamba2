from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as e2_cfg


class cfg(e2_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Image-level score is derived from top-k foreground anomaly-map responses.
        self.model.kwargs.setdefault('image_branch_kwargs', {})
        self.model.kwargs['image_branch_kwargs'].update(
            image_score_source='map_topk',
            map_topk_ratio=0.01,
        )
