from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_synthetic_brain_soft_e25 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Experiment C:
        # 1) Direction-consistent anomaly score: high score means closer to the
        #    abnormal prompt than the normal prompt.
        # 2) Decouple adapter conditioning from anomaly scoring. CSSD/AdaLN uses
        #    a fixed medical-domain condition in train and test, while scoring
        #    still uses good during auxiliary training and brain/liver/retinal
        #    prompts during target-domain test.
        self.anomaly_score_direction = 'abnormal_minus_normal'
        self.adapter_cls_name = 'generic_medical'
        self.fixed_adapter_cls_name = self.adapter_cls_name

        self.class_prompts.update({
            'generic_medical': 'A clinical medical image showing anatomical tissue structure.',
        })

        self.model.kwargs['class_prompts'] = self.class_prompts
        self.model.kwargs['anomaly_score_direction'] = self.anomaly_score_direction
        self.model.kwargs['adaptive_mc_kwargs']['anomaly_score_direction'] = self.anomaly_score_direction
