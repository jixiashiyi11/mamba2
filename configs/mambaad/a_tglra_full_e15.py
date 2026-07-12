from configs.mambaad.a_tglra_no_mamba_e15 import cfg as tglra_no_mamba_cfg


class cfg(tglra_no_mamba_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.name = 'mambaad_biomedclip_tglra_full'
        self.model.kwargs.update(
            image_branch_kwargs=dict(
                topk_ratio=0.05,
                image_score_beta=0.25,
                loss_weight=0.1,
                use_cssd=True,
            ),
            fusion_kwargs=dict(
                hidden_dim=512,
                dropout=0.0,
                residual_scale_init=0.1,
            ),
        )
