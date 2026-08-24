import copy

from configs.clip_ad.clip_ad_supervised_mask_pdar_cssd import cfg as base_cfg


class cfg(base_cfg):
    """V18 complete config: V13 behavior without Mamba separation loss."""

    def __init__(self):
        super().__init__()

        # ------------------------- data protocol -------------------------
        # Source domain: supervised MVTec images and anomaly masks.
        self.data_train = copy.deepcopy(self.data)
        self.data_train.type = "DefaultAD"
        self.data_train.root = "data/mvtec"
        self.data_train.meta = "meta_supervised.json"
        self.data_train.require_meta = True
        self.data_train.cls_names = []
        self.data_train.train_with_anomaly_masks = True
        self.data_train.enforce_disjoint_train_test = False

        # Target domain: VisA evaluation only.
        self.data_test = copy.deepcopy(self.data)
        self.data_test.type = "DefaultAD"
        self.data_test.root = "data"
        self.data_test.meta = "visa_meta.json"
        self.data_test.require_meta = True
        self.data_test.preserve_tiny_masks = True
        self.data_test.require_nonempty_anomaly_mask = True
        self.data_test.cls_names = [
            "pcb1", "pcb2", "pcb3", "pcb4",
            "macaroni1", "macaroni2", "capsules", "candle",
            "cashew", "chewinggum", "fryum", "pipe_fryum",
        ]
        self.data_test.enforce_disjoint_train_test = True

        # BaseTrainer is initialized with the source data namespace; the
        # CLIP-AD trainer replaces its test loader with data_test.
        self.data = copy.deepcopy(self.data_train)

        # ------------------------- PDAR / PVSS ---------------------------
        mamba_context_kwargs = dict(self.model.kwargs["mamba_context_kwargs"])
        mamba_context_kwargs.update(
            cssd_type="pdar",
            use_selective_scan=True,
            use_cnn_branch=True,
            use_deformable_pool=False,
            local_receptive_field_schedule=(
                (3, 5),
                (5, 7),
                (7, 9),
                (9, 11),
            ),
        )

        # Complete V18 model configuration. These values consolidate the
        # effective V13 settings previously inherited through V2/V3/V4/V9/V12.
        self.model.kwargs.update(
            mamba_context_kwargs=mamba_context_kwargs,

            # Source mask supervision shared with the preceding experiments.
            supervised_mask_bce_weight=1.5,
            supervised_mask_dice_weight=2.0,
            supervised_outside_topk_weight=0.1,
            supervised_image_weight=1.0,

            # Preserve tiny anomalies when supervising the 24x24 Mamba map.
            mamba_context_mask_pool="adaptive_max",

            # Train the independent Mamba verifier from source-domain masks.
            mamba_context_bce_weight=1.0,
            mamba_context_dice_weight=1.0,
            mamba_context_outside_topk_weight=0.1,
            # V18 ablation: disable the V17-only separation objective.
            mamba_context_separation_weight=0.0,
            mamba_context_separation_margin=0.0,

            # Keep the Mamba-verifier path required by evidence-MIL and ARCC
            # guidance, but disable its post-ARCC probability veto.
            arcc_mode="mamba_veto",
            mamba_veto_alpha_init=0.1,
            mamba_veto_temperature=1.0,
            mamba_veto_threshold=0.0,
            mamba_veto_detach=True,
            mamba_veto_enabled=False,

            # Feed only the detached single-channel Mamba support probability
            # into ARCC. The full Mamba feature is not injected.
            arcc_inject_mamba=False,
            arcc_mamba_support_guidance=True,
            arcc_mamba_support_detach=True,

            # Preserve V9/V13 seven-signal image-level evidence aggregation.
            image_score_mode="evidence_mil",
            image_score_topk_ratio=None,
        )

        # Replace only the external Layer-12/18 spatial CNN fusion with an
        # independent linear projection for every patch. PVSS internal CNN
        # branches remain enabled by the base configuration.
        adapter_kwargs = dict(self.model.kwargs["adapter_kwargs"])
        adapter_kwargs["fusion_mode"] = "projection"
        self.model.kwargs["adapter_kwargs"] = adapter_kwargs

        self.logging.log_terms_train.extend(
            [
                dict(name="loss_mamba_context_bce", fmt=":>5.3f", add_name="avg"),
                dict(name="loss_mamba_context_dice", fmt=":>5.3f", add_name="avg"),
                dict(name="loss_mamba_context_outside_topk", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_semantic_mean", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_semantic_max", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_support_mean", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_veto_mean", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_veto_alpha", fmt=":>5.3f", add_name="avg"),
                dict(name="dbg_mamba_veto_max_gain", fmt=":>7.5f", add_name="avg"),
                dict(name="dbg_image_fusion_w_global", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_raw_max", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_raw_top1", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_raw_top5", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_mamba_max", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_mamba_top1", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_w_mamba_top5", fmt=":>6.3f", add_name="avg"),
                dict(name="dbg_image_fusion_bias", fmt=":>6.3f", add_name="avg"),
                dict(
                    name="dbg_arcc_mamba_support_guidance_mean",
                    fmt=":>6.3f",
                    add_name="avg",
                ),
            ]
        )

        self.trainer.save_mamba_full_maps = False
        self.trainer.logdir_simple = True
        self.trainer.output_name = "outputs.npz"
        self.trainer.logdir_sub = (
            "pdar_mvtec_supervised_to_visa_v18_complete_v13"
        )
