from configs.mambaad.mambaad_medical_aux_train_balanced_loss_B_cons_0p1_biomedclip_dual_branch_text_e15 import cfg as dual_branch_cfg


class cfg(dual_branch_cfg):
    def __init__(self):
        super(cfg, self).__init__()

        image_branch_kwargs = dict(self.model.kwargs.get('image_branch_kwargs', {}))
        image_branch_kwargs['use_cssd'] = False
        self.model.kwargs['image_branch_kwargs'] = image_branch_kwargs
