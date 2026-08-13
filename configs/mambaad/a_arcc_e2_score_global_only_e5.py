from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as e2_cfg


class cfg(e2_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Image-level score comes only from global BiomedCLIP image-text similarity.
        self.model.kwargs.setdefault('image_branch_kwargs', {})
        self.model.kwargs['image_branch_kwargs'].update(
            image_score_source='global_only',
            image_score_beta=0.0,
        )
