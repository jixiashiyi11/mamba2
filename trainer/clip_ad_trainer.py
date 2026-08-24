import glob
import os
import shutil
import time

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from . import TRAINER
from ._base_trainer import BaseTrainer
from util.net import get_timepc, reduce_tensor
from util.util import able, log_msg, update_log_term


@TRAINER.register_module
class CLIPADTrainer(BaseTrainer):
    def __init__(self, cfg):
        cross_domain = hasattr(cfg, "data_train") and hasattr(cfg, "data_test")
        if cross_domain:
            # Let data.get_loader construct the source-train and target-test
            # loaders correctly during BaseTrainer initialization. This also
            # keeps the train/test lengths printed by log_cfg truthful.
            cfg.clip_ad_cross_domain = True
        super(CLIPADTrainer, self).__init__(cfg)
        if cross_domain:
            self._assert_cross_domain_disjoint()
            log_msg(
                self.logger,
                "==> Cross-domain mode: "
                f"source train root={cfg.data_train.root}, meta={cfg.data_train.meta}, "
                f"samples={self.train_loader.dataset.length}, batches={len(self.train_loader)}; "
                f"target test root={cfg.data_test.root}, meta={cfg.data_test.meta}, "
                f"samples={self.test_loader.dataset.length}, batches={len(self.test_loader)}",
            )
        self.cls_names = list(self.test_loader.dataset.cls_names)
        self._sync_metric_recorder(self.cls_names)
        log_msg(self.logger, f"==> Source-domain train classes: {list(self.train_loader.dataset.cls_names)}")
        log_msg(self.logger, f"==> Target-domain test classes: {self.cls_names}")

    @staticmethod
    def _dataset_image_paths(dataset):
        root = os.path.realpath(str(dataset.root))
        return {
            os.path.realpath(os.path.join(root, str(sample["img_path"])))
            for sample in dataset.samples
            if str(sample.get("img_path", "")).strip()
        }

    def _assert_cross_domain_disjoint(self):
        source_paths = self._dataset_image_paths(self.train_loader.dataset)
        target_paths = self._dataset_image_paths(self.test_loader.dataset)
        overlap = sorted(source_paths & target_paths)
        if overlap:
            examples = ", ".join(overlap[:5])
            raise RuntimeError(
                "Cross-domain data leakage detected: "
                f"{len(overlap)} image(s) occur in both source train and target test. "
                f"Examples: {examples}"
            )
        log_msg(
            self.logger,
            "==> Cross-domain leakage check passed: "
            f"source_train={len(source_paths)}, target_test={len(target_paths)}, overlap=0",
        )

    def _sync_metric_recorder(self, cls_names):
        existing = getattr(self, "metric_recorder", {}) or {}
        synced = {}
        for idx, cls_name in enumerate(cls_names):
            for metric in self.metrics:
                key = f"{metric}_{cls_name}"
                synced[key] = list(existing.get(key, []))
                if idx == len(cls_names) - 1 and len(cls_names) > 1:
                    avg_key = f"{metric}_Avg"
                    synced[avg_key] = list(existing.get(avg_key, []))
        self.metric_recorder = synced
        self.cfg.trainer.metric_recorder = synced

    @staticmethod
    def _pdar_region_means(weights, masks, labels):
        """Return equal-per-image PDAR depth means for three patch regions."""
        weights = weights.detach().float()
        masks = masks.detach().float()
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        # A 24x24 token is considered anomalous when any source-mask pixel in
        # its receptive bin is positive. Max pooling avoids losing tiny defects
        # during the 336x336 -> 24x24 reduction.
        masks = F.adaptive_max_pool2d(masks, output_size=weights.shape[-2:])[:, 0] > 0.5
        labels = labels.detach().reshape(-1).to(device=weights.device) != 0

        batch_size, num_sources = weights.shape[:2]
        nan = float("nan")
        normal_means = weights.new_full((batch_size, num_sources), nan)
        mask_in_means = weights.new_full((batch_size, num_sources), nan)
        mask_out_means = weights.new_full((batch_size, num_sources), nan)

        spatial_means = weights.mean(dim=(2, 3))
        normal_rows = ~labels
        normal_means[normal_rows] = spatial_means[normal_rows]

        for region_mask, destination in (
            (masks, mask_in_means),
            (~masks, mask_out_means),
        ):
            pixel_count = region_mask.sum(dim=(1, 2))
            valid_rows = labels & (pixel_count > 0)
            weighted_sum = (weights * region_mask.unsqueeze(1)).sum(dim=(2, 3))
            region_means = weighted_sum / pixel_count.clamp_min(1).unsqueeze(1)
            destination[valid_rows] = region_means[valid_rows]

        return {
            "normal": normal_means.cpu().numpy(),
            "mask_in": mask_in_means.cpu().numpy(),
            "mask_out": mask_out_means.cpu().numpy(),
        }

    def _pdar_region_table(self, results):
        rows = []
        block_names = [f"stage{idx}" for idx in range(1, 5)] + ["final"]
        for block_name in block_names:
            for region_name in ("normal", "mask_in", "mask_out"):
                key = f"mamba_depth_{block_name}_{region_name}_means"
                if key not in results:
                    continue
                values = np.asarray(results[key], dtype=np.float64)
                valid = np.isfinite(values).all(axis=1)
                if not np.any(valid):
                    continue
                means = values[valid].mean(axis=0)
                source_values = [f"{value:.6f}" for value in means]
                source_values.extend([""] * (5 - len(source_values)))
                rows.append([block_name, region_name, int(valid.sum()), *source_values])
        if rows:
            table = tabulate.tabulate(
                rows,
                headers=["Block", "Region", "Images", "F0", "F1", "F2", "F3", "F4"],
                tablefmt="pipe",
                stralign="center",
                numalign="center",
            )
            log_msg(self.logger, "==> PDAR patch-region depth weights\n" + table)

    def _expand_cls_name_like_batch(self, cls_names, batch_size):
        if isinstance(cls_names, str):
            return [cls_names] * batch_size
        if isinstance(cls_names, (list, tuple)):
            if len(cls_names) == batch_size:
                return list(cls_names)
            if len(cls_names) == 1:
                return list(cls_names) * batch_size
        return cls_names

    def _get_model_cls_names(self):
        score_cls_names = self.cls_name
        force_cls_name = getattr(self.cfg, "eval_force_cls_name", None)
        if not self.net.training and force_cls_name:
            score_cls_names = force_cls_name
        return self._expand_cls_name_like_batch(score_cls_names, self.bs)

    def set_input(self, inputs):
        self.imgs = inputs["img"].cuda()
        self.imgs_mask = inputs["img_mask"].cuda()
        self.cls_name = inputs["cls_name"]
        self.anomaly = inputs["anomaly"].cuda().long().view(-1)
        self.img_path = inputs.get("img_path") if isinstance(inputs, dict) else None
        self.mask_path = inputs.get("mask_path") if isinstance(inputs, dict) else None
        self.bs = self.imgs.shape[0]

    def forward(self, return_loss=True):
        score_cls_names = self._get_model_cls_names()
        self.output = self.net(
            self.imgs,
            cls_names=score_cls_names,
            masks=self.imgs_mask,
            labels=self.anomaly,
            return_loss=return_loss,
            # Trainer epochs are zero-based while the requested warmup
            # schedule is one-based: epoch 1 -> 0.1, epoch 2 -> 0.2.
            current_epoch=(self.epoch + 1 if return_loss else None),
        )
        if isinstance(self.output, tuple):
            self.anomaly_map, self.image_score = self.output
        else:
            self.anomaly_map = self.output["anomaly_map"]
            self.image_score = self.output["image_score"]
            if return_loss:
                self.total_loss = self.output["total"]

    def optimize_parameters(self):
        with self.amp_autocast():
            self.forward(return_loss=True)
            total_loss = self.total_loss
        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Non-finite total loss: {float(total_loss.detach().cpu())}")
        self.backward_term(total_loss, self.optim)
        for name in [
            "total",
            "loss_global",
            "loss_patch",
            "loss_normal_topk",
            "loss_consistency",
            "loss_image_normal",
            "loss_arcc_normal",
            "loss_arcc_cal",
            "loss_mask_bce",
            "loss_mask_dice",
            "loss_mask_raw_bce",
            "loss_outside_topk",
            "loss_image_supervised",
            "loss_pdar_image",
            "loss_mamba_context_bce",
            "loss_mamba_context_dice",
            "loss_mamba_context_outside_topk",
            "loss_mamba_context_separation",
            "loss_mamba_feature_contrast",
            "dbg_mamba_feature_contrast_weight",
            "dbg_mamba_feature_contrast_gap",
            "dbg_mamba_feature_prototype_cosine",
            "dbg_refine_cos",
            "dbg_refine_delta_l2",
            "dbg_mamba_context_cos",
            "dbg_mamba_context_delta_l2",
            "dbg_local_delta_l2",
            "dbg_local_delta_abs",
            "dbg_l12_last_cos",
            "dbg_l18_last_cos",
            "dbg_a_raw_mean",
            "dbg_a_raw_max",
            "dbg_a_final_mean",
            "dbg_a_final_max",
            "dbg_arcc_delta_abs",
            "dbg_arcc_delta_ratio",
            "dbg_arcc_lambda",
            "dbg_arcc_lambda_learned",
            "dbg_g_cal_abs",
            "dbg_arcc_mamba_injection_gamma",
            "dbg_arcc_cross_gamma",
            "dbg_arcc_context_gate_mean",
            "dbg_arcc_context_gate_max",
            "dbg_arcc_cross_feature_norm",
            "dbg_arcc_local_context_difference",
            "dbg_arcc_refine_step1_abs",
            "dbg_arcc_refine_step2_abs",
            "dbg_arcc_refine_step1_signed_mean",
            "dbg_arcc_refine_step2_signed_mean",
            "dbg_arcc_final_minus_raw_abs",
            "dbg_arcc_final_minus_raw_signed_mean",
            "dbg_arcc_dynamic_gate_norm",
            "dbg_arcc_delta_inside",
            "dbg_arcc_delta_outside",
            "dbg_arcc_gate_inside",
            "dbg_arcc_gate_outside",
            "dbg_arcc_normal_topk_before",
            "dbg_arcc_normal_topk_after",
            "dbg_mask_bce_normal",
            "dbg_mask_bce_abnormal",
            "dbg_mask_dice_positive",
            "dbg_normal_prob_mean",
            "dbg_normal_fg_ratio",
            "dbg_arcc_normal_max_gain",
            "dbg_raw_mask_gap",
            "dbg_final_mask_gap",
            "dbg_batch_normal_count",
            "dbg_batch_abnormal_count",
            "dbg_batch_positive_mask_count",
            "dbg_s_global",
            "dbg_topk_score",
            "dbg_topk_score_max",
            "dbg_topk_score_top1",
            "dbg_topk_score_top5",
            "dbg_image_score_max",
            "dbg_image_score_top1",
            "dbg_image_score_top5",
            "dbg_pdar_image_score_mean",
            "dbg_pdar_image_scale",
            "dbg_pdar_pool_entropy",
            "dbg_image_fusion_w_global",
            "dbg_image_fusion_w_raw_max",
            "dbg_image_fusion_w_raw_top1",
            "dbg_image_fusion_w_raw_top5",
            "dbg_image_fusion_w_mamba_max",
            "dbg_image_fusion_w_mamba_top1",
            "dbg_image_fusion_w_mamba_top5",
            "dbg_image_fusion_bias",
            "dbg_image_reviewer_w_global",
            "dbg_image_reviewer_w_raw",
            "dbg_image_reviewer_w_agree",
            "dbg_image_reviewer_w_disagree",
            "dbg_image_reviewer_bias",
            "dbg_image_reviewer_raw",
            "dbg_image_reviewer_mamba",
            "dbg_image_reviewer_agree",
            "dbg_image_reviewer_disagree",
            "dbg_image_reviewer_raw_mamba_corr",
            "dbg_image_reviewer_global_contrib",
            "dbg_image_reviewer_raw_contrib",
            "dbg_image_reviewer_agree_contrib",
            "dbg_image_reviewer_disagree_contrib",
            "dbg_reviewer_w_global",
            "dbg_reviewer_w_raw_max",
            "dbg_reviewer_w_raw_top1",
            "dbg_reviewer_w_raw_top5",
            "dbg_reviewer_bias",
            "dbg_reviewer_candidate_support",
            "dbg_reviewer_background_support",
            "dbg_reviewer_relative_gap",
            "dbg_reviewer_agree",
            "dbg_reviewer_reject",
            "dbg_reviewer_neutral_ratio",
            "dbg_reviewer_agree_scale",
            "dbg_reviewer_reject_scale",
            "dbg_reviewer_score_delta",
            "dbg_reviewer_base_score",
            "dbg_mamba_prior_mean",
            "dbg_mamba_prior_max",
            "dbg_mamba_prior_mask_in",
            "dbg_mamba_prior_mask_out",
            "dbg_mamba_prior_gap",
            "dbg_mamba_prior_normal_topk",
            "dbg_mamba_prior_abnormal_topk",
            "dbg_mamba_semantic_mean",
            "dbg_mamba_semantic_max",
            "dbg_mamba_verifier_mean",
            "dbg_mamba_verifier_max",
            "dbg_mamba_support_mean",
            "dbg_mamba_veto_mean",
            "dbg_mamba_veto_alpha",
            "dbg_mamba_veto_max_gain",
            "dbg_mamba_support_normal",
            "dbg_mamba_support_mask_in",
            "dbg_mamba_support_mask_out",
            "dbg_mamba_veto_normal",
            "dbg_mamba_veto_mask_in",
            "dbg_mamba_veto_mask_out",
            "dbg_mamba_depth_entropy",
            "dbg_mamba_depth_max_weight",
            "dbg_mamba_depth_w_f0",
            "dbg_mamba_depth_w_f1",
            "dbg_mamba_depth_w_f2",
            "dbg_mamba_depth_w_f3",
            "dbg_mamba_depth_w_f4",
            "dbg_mamba_s1_w_f0",
            "dbg_mamba_s2_w_f0",
            "dbg_mamba_s2_w_f1",
            "dbg_mamba_s3_w_f0",
            "dbg_mamba_s3_w_f1",
            "dbg_mamba_s3_w_f2",
            "dbg_mamba_s4_w_f0",
            "dbg_mamba_s4_w_f1",
            "dbg_mamba_s4_w_f2",
            "dbg_mamba_s4_w_f3",
        ]:
            value = total_loss if name == "total" else self.output.get(name)
            log_term = self.log_terms.get(name)
            if value is not None and log_term is not None:
                update_log_term(
                    log_term,
                    reduce_tensor(value, self.world_size).clone().detach().item(),
                    1,
                    self.master,
                )

    def _metric_table(self, results):
        msg = {}
        for idx, cls_name in enumerate(self.cls_names):
            metric_results = self.evaluator.run(results, cls_name, self.logger)
            msg["Name"] = msg.get("Name", [])
            msg["Name"].append(cls_name)
            avg_act = len(self.cls_names) > 1 and idx == len(self.cls_names) - 1
            msg["Name"].append("Avg") if avg_act else None
            for metric in self.metrics:
                metric_result = metric_results[metric] * 100
                self.metric_recorder[f"{metric}_{cls_name}"].append(metric_result)
                max_metric = max(self.metric_recorder[f"{metric}_{cls_name}"])
                max_metric_idx = self.metric_recorder[f"{metric}_{cls_name}"].index(max_metric) + 1
                msg[metric] = msg.get(metric, [])
                msg[metric].append(metric_result)
                msg[f"{metric} (Max)"] = msg.get(f"{metric} (Max)", [])
                msg[f"{metric} (Max)"].append(f"{max_metric:.3f} ({max_metric_idx:<3d} epoch)")
                if avg_act:
                    metric_result_avg = sum(msg[metric]) / len(msg[metric])
                    self.metric_recorder[f"{metric}_Avg"].append(metric_result_avg)
                    max_metric = max(self.metric_recorder[f"{metric}_Avg"])
                    max_metric_idx = self.metric_recorder[f"{metric}_Avg"].index(max_metric) + 1
                    msg[metric].append(metric_result_avg)
                    msg[f"{metric} (Max)"].append(f"{max_metric:.3f} ({max_metric_idx:<3d} epoch)")
        table = tabulate.tabulate(msg, headers="keys", tablefmt="pipe", floatfmt=".3f", numalign="center", stralign="center")
        log_msg(self.logger, f"\n{table}")

    def _image_score_variant_table(self, results):
        variant_keys = [
            ("default", "image_scores"),
            ("legacy", "image_scores_legacy"),
            ("pdar_only", "image_scores_pdar_only"),
            ("max", "image_scores_max"),
            ("top1", "image_scores_top1"),
            ("top5", "image_scores_top5"),
            ("cnn_only", "image_scores_cnn_only"),
        ]
        variant_keys = [(name, key) for name, key in variant_keys if key in results]
        if len(variant_keys) <= 1:
            return
        msg = {"Name": [], "Score": [], "AUROC_sp": [], "AP_sp": [], "F1max_sp": []}
        eps = 1e-8
        for cls_name in self.cls_names:
            idxes = results["cls_names"] == cls_name
            labels = results["anomalys"][idxes].reshape(-1).astype(int)
            for variant_name, key in variant_keys:
                scores = results[key][idxes].reshape(-1)
                if len(np.unique(labels)) < 2:
                    auroc = np.nan
                else:
                    auroc = roc_auc_score(labels, scores) * 100
                ap = average_precision_score(labels, scores) * 100
                precision, recall, _ = precision_recall_curve(labels, scores)
                f1 = (2.0 * precision * recall / (precision + recall + eps)).max() * 100
                msg["Name"].append(cls_name)
                msg["Score"].append(variant_name)
                msg["AUROC_sp"].append(auroc)
                msg["AP_sp"].append(ap)
                msg["F1max_sp"].append(f1)
        table = tabulate.tabulate(msg, headers="keys", tablefmt="pipe", floatfmt=".3f", numalign="center", stralign="center")
        log_msg(self.logger, f"\n==> Image score aggregation variants\n{table}")

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        force_cls_name = getattr(self.cfg, "eval_force_cls_name", None)
        if self.master and force_cls_name:
            log_msg(self.logger, f"==> EvalForceClsName score_cls_name={force_cls_name} metric_cls_names=original")

        imgs_masks, anomaly_maps, image_scores, cls_names, anomalys, layer_text_maps = [], [], [], [], [], []
        image_scores_max, image_scores_top1, image_scores_top5 = [], [], []
        raw_anomaly_maps, arcc_cal_maps, mamba_prior_maps = [], [], []
        mamba_semantic_maps, mamba_support_maps, mamba_veto_maps = [], [], []
        save_mamba_full_maps = bool(getattr(self.cfg.trainer, "save_mamba_full_maps", True))
        image_component_outputs = {
            "global_scores": "S_global",
            "raw_scores_max": "raw_score_max",
            "raw_scores_top1": "raw_score_top1",
            "raw_scores_top5": "raw_score_top5",
            "mamba_scores_max": "mamba_score_max",
            "mamba_scores_top1": "mamba_score_top1",
            "mamba_scores_top5": "mamba_score_top5",
            "image_scores_raw_top5": "image_score_raw_top5",
            "image_scores_mamba_top5": "image_score_mamba_top5",
            "image_scores_cnn_only": "image_score_cnn_only",
            "image_scores_legacy": "image_score_legacy",
            "image_scores_pdar_only": "image_score_pdar_only",
            "image_evidence": "image_evidence",
            "image_reviewer_evidence": "image_reviewer_evidence",
            "image_relative_reviewer_evidence": "image_relative_reviewer_evidence",
            "image_scores_relative_reviewer_base": "image_score_relative_reviewer_base",
        }
        image_component_values = {key: [] for key in image_component_outputs}
        mamba_depth_stage_means = {}
        mamba_depth_region_means = {}
        diagnostic_specs = {
            "dbg_mask_bce_normal": "dbg_batch_normal_count",
            "dbg_mask_bce_abnormal": "dbg_batch_abnormal_count",
            "dbg_mask_dice_positive": "dbg_batch_positive_mask_count",
            "dbg_normal_prob_mean": "dbg_batch_normal_count",
            "dbg_normal_fg_ratio": "dbg_batch_normal_count",
            "dbg_arcc_normal_max_gain": "dbg_batch_normal_count",
            "dbg_raw_mask_gap": "dbg_batch_positive_mask_count",
            "dbg_final_mask_gap": "dbg_batch_positive_mask_count",
            "dbg_mamba_support_normal": "dbg_batch_normal_count",
            "dbg_mamba_support_mask_in": "dbg_batch_positive_mask_count",
            "dbg_mamba_support_mask_out": "dbg_batch_positive_mask_count",
            "dbg_mamba_veto_normal": "dbg_batch_normal_count",
            "dbg_mamba_veto_mask_in": "dbg_batch_positive_mask_count",
            "dbg_mamba_veto_mask_out": "dbg_batch_positive_mask_count",
            "dbg_joint_vs_cnn_max_normal": "dbg_batch_normal_count",
            "dbg_joint_vs_cnn_max_abnormal": "dbg_batch_abnormal_count",
            "dbg_joint_vs_cnn_mask_in": "dbg_batch_positive_mask_count",
            "dbg_joint_vs_cnn_mask_out": "dbg_batch_positive_mask_count",
        }
        diagnostic_sums = {key: 0.0 for key in diagnostic_specs}
        diagnostic_counts = {key: 0.0 for key in diagnostic_specs}
        img_paths, mask_paths = [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward(return_loss=False)
            if isinstance(self.output, dict):
                for key, count_key in diagnostic_specs.items():
                    value = self.output.get(key)
                    count = self.output.get(count_key)
                    if value is None or count is None:
                        continue
                    count_value = float(count.detach().cpu())
                    diagnostic_sums[key] += float(value.detach().cpu()) * count_value
                    diagnostic_counts[key] += count_value
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(self.anomaly_map.cpu().numpy())
            image_scores.append(self.image_score.cpu().numpy())
            if isinstance(self.output, dict) and "image_score_max" in self.output:
                image_scores_max.append(self.output["image_score_max"].cpu().numpy())
            if isinstance(self.output, dict) and "image_score_top1" in self.output:
                image_scores_top1.append(self.output["image_score_top1"].cpu().numpy())
            if isinstance(self.output, dict) and "image_score_top5" in self.output:
                image_scores_top5.append(self.output["image_score_top5"].cpu().numpy())
            if isinstance(self.output, dict):
                for result_key, output_key in image_component_outputs.items():
                    if output_key in self.output:
                        image_component_values[result_key].append(
                            self.output[output_key].cpu().numpy()
                        )
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))
            if isinstance(self.output, dict) and "layer_text_maps" in self.output:
                layer_text_maps.append(self.output["layer_text_maps"].cpu().numpy())
            if isinstance(self.output, dict) and "A_raw" in self.output:
                raw_anomaly_maps.append(self.output["A_raw"].cpu().numpy())
            if isinstance(self.output, dict) and "G_cal" in self.output:
                arcc_cal_maps.append(self.output["G_cal"].cpu().numpy())
            if isinstance(self.output, dict) and "mamba_global_prior" in self.output:
                mamba_prior_maps.append(self.output["mamba_global_prior"].cpu().numpy())
            if save_mamba_full_maps and isinstance(self.output, dict) and "mamba_semantic_map" in self.output:
                mamba_semantic_maps.append(self.output["mamba_semantic_map"].cpu().numpy())
            if save_mamba_full_maps and isinstance(self.output, dict) and "mamba_support_map" in self.output:
                mamba_support_maps.append(self.output["mamba_support_map"].cpu().numpy())
            if save_mamba_full_maps and isinstance(self.output, dict) and "mamba_veto_map" in self.output:
                mamba_veto_maps.append(self.output["mamba_veto_map"].cpu().numpy())
            if isinstance(self.output, dict) and "mamba_depth_stage_weight_means" in self.output:
                for stage_idx, stage_means in enumerate(
                    self.output["mamba_depth_stage_weight_means"], start=1
                ):
                    key = f"mamba_depth_stage{stage_idx}_means"
                    mamba_depth_stage_means.setdefault(key, []).append(stage_means.cpu().numpy())
            if isinstance(self.output, dict) and "mamba_depth_final_weight_means" in self.output:
                mamba_depth_stage_means.setdefault("mamba_depth_final_means", []).append(
                    self.output["mamba_depth_final_weight_means"].cpu().numpy()
                )
            if isinstance(self.output, dict) and "mamba_depth_stage_weights" in self.output:
                depth_blocks = [
                    (f"stage{stage_idx}", stage_weights)
                    for stage_idx, stage_weights in enumerate(
                        self.output["mamba_depth_stage_weights"], start=1
                    )
                ]
                depth_blocks.append(("final", self.output["mamba_depth_weights"]))
                for block_name, weights in depth_blocks:
                    region_means = self._pdar_region_means(
                        weights, self.imgs_mask, self.anomaly
                    )
                    for region_name, values in region_means.items():
                        key = f"mamba_depth_{block_name}_{region_name}_means"
                        mamba_depth_region_means.setdefault(key, []).append(values)
            if self.img_path is not None:
                img_paths.append(np.array(self.img_path))
            if self.mask_path is not None:
                mask_paths.append(np.array(self.mask_path))
            t2 = get_timepc()
            update_log_term(self.log_terms.get("batch_t"), t2 - t1, 1, self.master)
            print(f"\r{batch_idx}/{test_length}", end="") if self.master else None
            if self.master and (batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length):
                msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix="Test"), self.master, None)
                log_msg(self.logger, msg)

        diagnostic_pairs = torch.tensor(
            [
                value
                for key in diagnostic_specs
                for value in (diagnostic_sums[key], diagnostic_counts[key])
            ],
            device=self.imgs.device,
            dtype=torch.float64,
        )
        if self.cfg.dist:
            torch.distributed.all_reduce(diagnostic_pairs, op=torch.distributed.ReduceOp.SUM)
        if self.master:
            diagnostic_rows = []
            pair_values = diagnostic_pairs.cpu().numpy().reshape(-1, 2)
            for (key, _), (weighted_sum, count) in zip(
                diagnostic_specs.items(),
                pair_values,
            ):
                mean = weighted_sum / count if count > 0 else float("nan")
                diagnostic_rows.append([key, f"{mean:.6f}", int(count)])
            diagnostic_table = tabulate.tabulate(
                diagnostic_rows,
                headers=["Diagnostic", "Mean", "Images"],
                tablefmt="pipe",
                stralign="left",
                numalign="right",
            )
            log_msg(self.logger, "==> Mask/ARCC diagnostic summary\n" + diagnostic_table)

        results = dict(
            imgs_masks=imgs_masks,
            anomaly_maps=anomaly_maps,
            image_scores=image_scores,
            cls_names=cls_names,
            anomalys=anomalys,
        )
        if img_paths:
            results["img_paths"] = img_paths
        if mask_paths:
            results["mask_paths"] = mask_paths
        if layer_text_maps:
            results["layer_text_maps"] = layer_text_maps
        if raw_anomaly_maps:
            results["raw_anomaly_maps"] = raw_anomaly_maps
        if arcc_cal_maps:
            results["arcc_cal_maps"] = arcc_cal_maps
        if mamba_prior_maps:
            results["mamba_prior_maps"] = mamba_prior_maps
        if mamba_semantic_maps:
            results["mamba_semantic_maps"] = mamba_semantic_maps
        if mamba_support_maps:
            results["mamba_support_maps"] = mamba_support_maps
        if mamba_veto_maps:
            results["mamba_veto_maps"] = mamba_veto_maps
        for key, values in image_component_values.items():
            if values:
                results[key] = values
        for key, values in mamba_depth_stage_means.items():
            results[key] = values
        for key, values in mamba_depth_region_means.items():
            results[key] = values
        if image_scores_max:
            results["image_scores_max"] = image_scores_max
        if image_scores_top1:
            results["image_scores_top1"] = image_scores_top1
        if image_scores_top5:
            results["image_scores_top5"] = image_scores_top5
        if self.cfg.dist:
            torch.save(results, f"{self.tmp_dir}/{self.rank}.pth", _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], image_scores=[], cls_names=[], anomalys=[])
                if img_paths:
                    results["img_paths"] = []
                if mask_paths:
                    results["mask_paths"] = []
                if layer_text_maps:
                    results["layer_text_maps"] = []
                if raw_anomaly_maps:
                    results["raw_anomaly_maps"] = []
                if arcc_cal_maps:
                    results["arcc_cal_maps"] = []
                if mamba_prior_maps:
                    results["mamba_prior_maps"] = []
                if mamba_semantic_maps:
                    results["mamba_semantic_maps"] = []
                if mamba_support_maps:
                    results["mamba_support_maps"] = []
                if mamba_veto_maps:
                    results["mamba_veto_maps"] = []
                for key, values in image_component_values.items():
                    if values:
                        results[key] = []
                for key in mamba_depth_stage_means:
                    results[key] = []
                for key in mamba_depth_region_means:
                    results[key] = []
                if image_scores_max:
                    results["image_scores_max"] = []
                if image_scores_top1:
                    results["image_scores_top1"] = []
                if image_scores_top5:
                    results["image_scores_top5"] = []
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f"{self.tmp_dir}/*.pth")
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            try:
                                result = torch.load(results_files[idx_result])
                                for key, value in result.items():
                                    results[key].extend(value)
                                idx_result += 1
                            except Exception:
                                time.sleep(1)
                        valid_results = True

        if self.master:
            results = {key: np.concatenate(value, axis=0) for key, value in results.items()}
            net_for_stage = getattr(self.net, "module", self.net)
            model_stage = getattr(net_for_stage, "stage", "stage1")
            output_name = getattr(self.cfg.trainer, "output_name", "")
            if not output_name:
                output_name = f"clip_ad_{model_stage}_outputs.npz"
            save_path = os.path.join(self.cfg.logdir_test, output_name)
            np.savez_compressed(save_path, **results)
            log_msg(self.logger, f"==> Saved CLIP AD outputs: {save_path}")
            self._pdar_region_table(results)
            self._image_score_variant_table(results)
            self._metric_table(results)
