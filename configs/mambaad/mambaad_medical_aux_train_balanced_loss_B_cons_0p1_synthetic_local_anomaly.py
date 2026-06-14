from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.synthetic_local_anomaly.enabled = True
        self.synthetic_local_anomaly.prob = 1.0
        self.synthetic_local_anomaly.loss_weight = 0.1
        self.synthetic_local_anomaly.bce_weight = 1.0
        self.synthetic_local_anomaly.dice_weight = 1.0
        self.synthetic_local_anomaly.score_temperature = 0.1
        self.synthetic_local_anomaly.min_area = 0.005
        self.synthetic_local_anomaly.max_area = 0.08
        self.synthetic_local_anomaly.noise_std = 0.18
        self.synthetic_local_anomaly.intensity_delta = 0.35
        self.synthetic_local_anomaly.foreground_threshold = 5.0 / 255.0

        self.model.kwargs['adaptive_mc_kwargs']['lambda_score_separation'] = 0.02
        self.model.kwargs['adaptive_mc_kwargs']['score_target'] = 0.0

        self.prompt_abnormal.update({
            'good': 'A medical image with a localized visible abnormal region or lesion.',
        })
        self.class_prompts.update({
            'good': 'A medical image used for learning localized abnormal region detection.',
        })
        self.model.kwargs['prompt_abnormal'] = self.prompt_abnormal
        self.model.kwargs['class_prompts'] = self.class_prompts
