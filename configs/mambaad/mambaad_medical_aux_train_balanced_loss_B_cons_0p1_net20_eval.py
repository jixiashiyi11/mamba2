from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.trainer.resume_dir = (
            'MAMBAADZeroShotTrainer_configs_mambaad_'
            'mambaad_medical_aux_train_balanced_loss_B_cons_0p1_B_cons_0p1_e25'
        )
        self.model.kwargs['checkpoint_path'] = 'net_20.pth'
