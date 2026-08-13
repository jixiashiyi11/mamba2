from configs.mambaad.a_arcc_e2_feature_calib_e5 import cfg as full_cfg


class cfg(full_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Base loss: keep only synthetic-mask BCE + Dice supervision.
        self.model.kwargs['local_loss_kwargs'].update(
            normal_topk_loss_weight=0.0,
            background_loss_weight=0.0,
            edge_loss_weight=0.0,
        )
        self.model.kwargs['image_branch_kwargs'].update(
            loss_weight=0.0,
        )
        self.model.kwargs['text_guidance_kwargs'].update(
            semantic_gate_loss_weight=0.0,
            prototype_reg_weight=0.0,
        )

        self.synthetic_local_anomaly.loss_weight = 0.1
        self.synthetic_local_anomaly.bce_weight = 1.0
        self.synthetic_local_anomaly.dice_weight = 1.0
        self.synthetic_local_anomaly.outside_suppression_weight = 0.0
        self.synthetic_local_anomaly.area_sparsity_weight = 0.0
