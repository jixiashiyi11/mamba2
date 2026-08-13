from configs.mambaad.a_arcc_e2_global_map_topk_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.trainer.resume_dir = 'MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333'
        self.model.kwargs['checkpoint_path'] = 'net.pth'
