from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_local_adapter_mixed_wavelet_mild_e15 import cfg as mild_cfg


class cfg(mild_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.synthetic_local_anomaly.outside_suppression_weight = 0.2
        self.synthetic_local_anomaly.area_sparsity_weight = 0.05
        self.synthetic_local_anomaly.outside_mask_dilate_iters = 2
        self.synthetic_local_anomaly.compact_mask_threshold = 0.05
        self.synthetic_local_anomaly.area_target_multiplier = 1.5
        self.synthetic_local_anomaly.area_target_slack = 0.005

        self.model.kwargs['local_loss_kwargs'].update(
            normal_topk_loss_weight=0.4,
            background_loss_weight=0.05,
            edge_loss_weight=0.05,
            normal_topk_ratio=0.01,
        )
