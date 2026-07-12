from configs.mambaad.a_cnn_global_aux_multiweak_e15 import cfg as multiweak_cfg


class cfg(multiweak_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.epoch_full = 5
        self.test_start_epoch = 5
        self.test_per_epoch = 5
        self.trainer.epoch_full = self.epoch_full
        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch
        self.trainer.scheduler_kwargs['decay_epochs'] = int(self.epoch_full * 0.8)

