from configs.mambaad.mambaad_medical_aux_train_balanced_loss_ablation import cfg as ablation_cfg


class cfg(ablation_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.model.kwargs['adaptive_mc_kwargs']['lambda_cons'] = 0.1
