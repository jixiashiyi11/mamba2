from configs.mambaad.a_cnn_global_aux_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.data.name = 'medical_aux_multisource_normal'
        self.data.root = 'data/medical_aux_multisource_normal'
        self.data.cls_names = ['good']

        self.data_train.name = self.data.name
        self.data_train.root = self.data.root
        self.data_train.cls_names = ['good']

        self.trainer.logdir_sub = 'multisource_normal_e5'

