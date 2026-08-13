from configs.mambaad.a_arcc_e3_response_calib_e5 import cfg as base_cfg


class cfg(base_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        # Prompt ensemble: keep the same text-prototype mechanism, but average
        # several normal/abnormal descriptions for more stable BiomedCLIP priors.
        self.prompt_normal.update({
            'good': [
                'A normal healthy medical image with no visible pathological abnormality.',
                'A disease-free medical image without visible lesion or abnormal region.',
                'A normal medical scan showing preserved anatomical structure and no pathology.',
            ],
            'brain': [
                'A normal healthy brain MRI scan with symmetric hemispheres and no visible lesion or structural abnormality.',
                'A disease-free brain MRI image without tumor, edema, hemorrhage, or infarct.',
                'A normal brain MRI scan showing preserved anatomy and no pathological finding.',
            ],
            'liver': [
                'A normal healthy liver CT scan with homogeneous parenchyma, clear boundaries, and no focal lesion.',
                'A disease-free liver CT image without metastasis, cyst, cirrhotic change, or focal abnormality.',
                'A normal liver CT scan showing preserved organ contour and homogeneous tissue appearance.',
            ],
            'retinal': [
                'A normal healthy retinal fundus photograph with a clear optic disc, macula, and no visible pathological finding.',
                'A disease-free retinal fundus image without hemorrhage, exudate, microaneurysm, or neovascularization.',
                'A normal retina image showing preserved vessels, optic disc, and macular region.',
            ],
        })
        self.prompt_abnormal.update({
            'good': [
                'A medical image with a localized visible abnormal region or lesion.',
                'A pathological medical image showing a focal abnormality or disease-related region.',
                'An abnormal medical scan containing a suspicious lesion area.',
            ],
            'brain': [
                'An abnormal brain MRI scan showing a lesion, tumor, hemorrhage, edema, infarct, or other pathological structural abnormality.',
                'A pathological brain MRI image with a localized abnormal region or lesion.',
                'An abnormal brain MRI scan containing suspicious disease-related tissue changes.',
            ],
            'liver': [
                'An abnormal liver CT scan showing a focal lesion, metastasis, cyst, cirrhotic change, or other pathological structural abnormality.',
                'A pathological liver CT image with a localized abnormal region or lesion.',
                'An abnormal liver CT scan containing suspicious focal tissue changes.',
            ],
            'retinal': [
                'An abnormal retinal fundus photograph showing hemorrhages, exudates, microaneurysms, neovascularization, or other pathological abnormality.',
                'A pathological retinal fundus image with visible disease-related lesions.',
                'An abnormal retina image containing suspicious hemorrhage, exudate, or vascular abnormality.',
            ],
        })
        self.model.kwargs['prompt_normal'] = self.prompt_normal
        self.model.kwargs['prompt_abnormal'] = self.prompt_abnormal

        # More lesion-like synthetic anomalies: smaller, softer, and less
        # aggressive than the strong default blobs.
        self.synthetic_local_anomaly.prob = 0.8
        self.synthetic_local_anomaly.loss_weight = 0.3
        self.synthetic_local_anomaly.min_area = 0.001
        self.synthetic_local_anomaly.max_area = 0.035
        self.synthetic_local_anomaly.noise_std = 0.06
        self.synthetic_local_anomaly.intensity_delta = 0.14
        self.synthetic_local_anomaly.foreground_erode_iters = 1
        self.synthetic_local_anomaly.num_blobs_min = 1
        self.synthetic_local_anomaly.num_blobs_max = 3
        self.synthetic_local_anomaly.soft_edge_power = 1.6
        self.synthetic_local_anomaly.soft_mask_threshold = 0.03

        self.model.kwargs['text_guidance_kwargs'].update(
            prototype_reg_weight=0.03,
        )
        self.model.kwargs['image_branch_kwargs'].update(
            topk_ratio=0.03,
            map_topk_ratio=0.01,
            image_score_beta=0.2,
            loss_weight=0.1,
        )
        self.model.kwargs['local_loss_kwargs'].update(
            normal_topk_ratio=0.01,
            normal_topk_loss_weight=0.25,
            background_loss_weight=0.05,
            edge_loss_weight=0.05,
        )
        self.model.kwargs['arcc_kwargs'].update(
            lambda_init=0.2,
            use_response=True,
            use_foreground=False,
            use_edge=False,
            kernel_size=3,
        )
