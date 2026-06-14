from configs.mambaad.mambaad_medical_aux_train_balanced import cfg as aux_cfg


class cfg(aux_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Baseline equals the current AdaptiveMCLoss behavior:
        # total = normal_align + 0.05 * adaptive_margin + 1.0 * token_consistency.
        # Ablations should override only these kwargs from the command line.
        self.model.kwargs['adaptive_mc_kwargs'] = dict(
            m_base=0.2,
            alpha=0.3,
            lambda_normal_align=1.0,
            lambda_margin=0.05,
            lambda_cons=1.0,
            lambda_score_separation=0.0,
            score_topk_ratio=0.1,
            score_temperature=0.1,
            score_target=0.0,
        )
