from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_net20_eval import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.eval_adapter_mode = 'trained'
        self.eval_force_cls_name = 'good'
        self.debug_eval = True
