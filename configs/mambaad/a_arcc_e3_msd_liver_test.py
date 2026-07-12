import copy

from configs.mambaad.a_arcc_e3_response_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.data_test = copy.deepcopy(self.data_test)
        self.data_test.type = 'DefaultAD'
        self.data_test.name = 'medical_standard_msd_liver'
        self.data_test.root = 'data/medical_standard_msd_liver'
        self.data_test.meta = 'meta.json'
        self.data_test.cls_names = ['liver']

        self.debug_eval = True
        self.debug_eval_vis_per_organ = 50
        self.debug_eval_save_anomalyclip_style = True
