from configs.mambaad.a_arcc_e0_baseline_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        image_branch_kwargs = dict(self.model.kwargs.get('image_branch_kwargs', {}))
        image_branch_kwargs.update(
            image_score_source='map_topk',
            map_topk_ratio=0.01,
        )
        self.model.kwargs['image_branch_kwargs'] = image_branch_kwargs
