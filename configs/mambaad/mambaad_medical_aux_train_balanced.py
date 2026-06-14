import copy

from configs.mambaad.mambaad_medical import cfg as medical_cfg


class cfg(medical_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        self.data_test = copy.deepcopy(self.data)
        self.data_test.type = 'DefaultAD'
        self.data_test.name = 'medical'
        self.data_test.root = 'data/medical'
        self.data_test.meta = 'meta.json'
        self.data_test.cls_names = []

        self.data.type = 'FolderNormalAD'
        self.data.name = 'medical_aux_train_balanced'
        self.data.root = 'data/medical_aux_train_balanced'
        self.data.cls_names = ['good']
        self.data_train = copy.deepcopy(self.data)

        self.prompt_normal.update({
            'good': 'A normal healthy medical image with no visible pathological abnormality.',
        })
        self.prompt_abnormal.update({
            'good': 'An abnormal medical image showing a visible pathological abnormality.',
        })
        self.class_prompts.update({
            'good': 'A medical image from the balanced normal training set.',
        })
        self.class_prompt_template = 'A medical image of {class_name}'
        self.model.kwargs['prompt_normal'] = self.prompt_normal
        self.model.kwargs['prompt_abnormal'] = self.prompt_abnormal
        self.model.kwargs['class_prompts'] = self.class_prompts
        self.model.kwargs['class_prompt_template'] = self.class_prompt_template
