from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_dual_branch_text_no_cssd_e15 import cfg as no_cssd_cfg


class cfg(no_cssd_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.epoch_full = 5
        self.test_start_epoch = 5
        self.test_per_epoch = 5
        self.trainer.epoch_full = self.epoch_full
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.scheduler_kwargs['decay_epochs'] = int(self.epoch_full * 0.8)
