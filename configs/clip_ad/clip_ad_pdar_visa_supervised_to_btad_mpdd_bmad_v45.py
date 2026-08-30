"""V45: train on labeled VisA and test BTAD, MPDD, and BMAD medical sets."""

import copy

from configs.clip_ad.clip_ad_pdar_visa_supervised_with_other_to_mvtec_v44 import (
    cfg as base_cfg,
)


class cfg(base_cfg):
    """Keep V44 training/model settings and replace only the target benchmark."""

    def __init__(self):
        super().__init__()

        self.data_test = copy.deepcopy(self.data_test)
        self.data_test.root = "data"
        self.data_test.meta = "visa_to_benchmarks_v45.json"
        self.data_test.require_meta = True
        self.data_test.cls_names = [
            "BTAD_01",
            "BTAD_02",
            "BTAD_03",
            "MPDD_bracket_black",
            "MPDD_bracket_brown",
            "MPDD_bracket_white",
            "MPDD_connector",
            "MPDD_metal_plate",
            "MPDD_tubes",
            "Brain_MRI",
            "Liver_CT",
            "Retina_OCT",
        ]
        self.data_test.preserve_tiny_masks = True
        self.data_test.require_nonempty_anomaly_mask = True
        self.data_test.enforce_disjoint_train_test = True

        # The ordinary metric table remains per category. This mapping adds the
        # dataset-level macro averages needed for comparison with paper tables.
        self.eval_class_groups = {
            "BTAD": ["BTAD_01", "BTAD_02", "BTAD_03"],
            "MPDD": [
                "MPDD_bracket_black",
                "MPDD_bracket_brown",
                "MPDD_bracket_white",
                "MPDD_connector",
                "MPDD_metal_plate",
                "MPDD_tubes",
            ],
            "Brain_MRI": ["Brain_MRI"],
            "Liver_CT": ["Liver_CT"],
            "Retina_OCT": ["Retina_OCT"],
        }

        # BaseTrainer reads self.data for sizes and logging.
        self.data = copy.deepcopy(self.data_train)

        self.trainer.resume_dir = ""
        self.model.kwargs["checkpoint_path"] = ""
        self.trainer.logdir_sub = (
            "pdar_visa_supervised_to_btad_mpdd_bmad_v45"
        )
        self.trainer.logdir_simple = True
        self.trainer.output_name = "outputs.npz"
