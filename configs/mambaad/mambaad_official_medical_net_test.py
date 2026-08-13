from configs.mambaad.mambaad_official_medical_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super().__init__()

        self.model.name = 'mambaadhcs57c1'
        self.model.kwargs['checkpoint_path'] = 'model/pretrain/mambaad_medical.pth'
        self.model.kwargs['strict'] = True

        self.trainer.name = 'MAMBAADOfficialRefTrainer'
        self.trainer.resume_dir = ''
        self.trainer.logdir_sub = 'official_weight_test'
        self.trainer.epoch_full = 0
        self.trainer.test_start_epoch = 0
        self.trainer.test_per_epoch = 1
        self.trainer.data.batch_size_per_gpu_test = 8
