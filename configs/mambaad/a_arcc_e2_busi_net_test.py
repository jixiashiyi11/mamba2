from configs.mambaad.a_arcc_e2_busi_test import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.trainer.resume_dir = 'MAMBAADZeroShotTrainer_configs_mambaad_a_arcc_e2_feature_calib_e5_20260711-222333'
        self.model.kwargs['checkpoint_path'] = 'net.pth'

        self.debug_eval = True
        self.debug_eval_vis_per_organ = 80
        self.debug_eval_save_anomalyclip_style = True
        self.debug_eval_anomalyclip_style_size = 512
        self.debug_eval_anomalyclip_style_draw_mask = True

        self.debug_eval_dirname = 'debug_eval_busi'
        self.debug_vis_dirname = 'debug_vis_busi'
