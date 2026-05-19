import os
import glob
import shutil
import time

import numpy as np
import tabulate
import torch
import torch.nn.functional as F

from util.util import able, log_msg, update_log_term
from util.net import get_timepc, reduce_tensor

from ._base_trainer import BaseTrainer
from . import TRAINER


@TRAINER.register_module
class MAMBAADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(MAMBAADTrainer, self).__init__(cfg)
        self.device = torch.device(f'cuda:{cfg.local_rank}')
        self.lambda_l1 = getattr(cfg.loss, 'lambda_l1', 0.005)
        self.adaptive_mc_weight_start = getattr(cfg.loss, 'adaptive_mc_weight_start', 0.01)
        self.adaptive_mc_weight_end = getattr(cfg.loss, 'adaptive_mc_weight_end', 1.0)
        self.adaptive_mc_warmup_epochs = getattr(cfg.loss, 'adaptive_mc_warmup_epochs', 0)
        self.class_loss_ema_momentum = getattr(cfg.loss, 'class_loss_ema_momentum', 0.9)
        self.class_weight_high = getattr(cfg.loss, 'class_weight_high', 1.5)
        self.class_weight_low = getattr(cfg.loss, 'class_weight_low', 0.8)
        self.use_adaptive_mc = 'adaptive_mc' in self.loss_terms
        self.balance_class_names = [str(name).lower() for name in self.cls_names]
        self.balance_class_to_idx = {name: idx for idx, name in enumerate(self.balance_class_names)}
        self.class_loss_ema = torch.ones(len(self.balance_class_names), device=self.device)
        self.prior_names = []
        self.prior_name_to_idx = {}
        self.T_norm_prior = None
        self.T_abn_prior = None
        if self.use_adaptive_mc:
            self.prior_names, self.T_norm_prior, self.T_abn_prior = self._setup_text_priors()
            self.prior_name_to_idx = {name: idx for idx, name in enumerate(self.prior_names)}

    def _normalize_prompt_config(self, prompt_config, name):
        if isinstance(prompt_config, str):
            return {'__shared__': prompt_config}
        if isinstance(prompt_config, dict):
            if not prompt_config:
                raise ValueError(f'`{name}` must not be an empty dict.')
            return {str(key).lower(): value for key, value in prompt_config.items()}
        raise TypeError(f'`{name}` must be a string or dict, got {type(prompt_config).__name__}.')

    def _resolve_prompt_template(self, prompt_template, cls_name):
        if '{cls_name}' not in prompt_template:
            return prompt_template
        return prompt_template.format(cls_name=cls_name)

    def _build_prompt_pairs(self, prompt_normal, prompt_abnormal):
        normal_map = self._normalize_prompt_config(prompt_normal, 'prompt_normal')
        abnormal_map = self._normalize_prompt_config(prompt_abnormal, 'prompt_abnormal')

        if '__shared__' in normal_map and '__shared__' in abnormal_map:
            cls_names = ['__shared__']
        elif '__shared__' in normal_map:
            cls_names = list(abnormal_map.keys())
            normal_map = {name: normal_map['__shared__'] for name in cls_names}
        elif '__shared__' in abnormal_map:
            cls_names = list(normal_map.keys())
            abnormal_map = {name: abnormal_map['__shared__'] for name in cls_names}
        else:
            cls_names = sorted(normal_map.keys())
            if set(cls_names) != set(abnormal_map.keys()):
                raise ValueError('`prompt_normal` and `prompt_abnormal` must have the same class keys.')

        normal_prompts = [self._resolve_prompt_template(normal_map[name], name) for name in cls_names]
        abnormal_prompts = [self._resolve_prompt_template(abnormal_map[name], name) for name in cls_names]
        return cls_names, normal_prompts, abnormal_prompts

    def _setup_text_priors(self):
        prompt_normal = getattr(self.cfg, 'prompt_normal', None)
        prompt_abnormal = getattr(self.cfg, 'prompt_abnormal', None)
        if not prompt_normal or not prompt_abnormal:
            raise ValueError('`prompt_normal` and `prompt_abnormal` must be defined in the medical config.')

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                'BiomedCLIP prior extraction requires the `open_clip` package to be installed.'
            ) from exc

        model_name = getattr(
            self.cfg,
            'biomedclip_model_name',
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        )
        cls_names, normal_prompts, abnormal_prompts = self._build_prompt_pairs(prompt_normal, prompt_abnormal)

        log_msg(self.logger, f'==> Encoding cached BiomedCLIP priors from {model_name}')
        text_encoder, _, _ = open_clip.create_model_and_transforms(model_name)
        tokenizer = open_clip.get_tokenizer(model_name)
        text_encoder = text_encoder.to(self.device)
        text_encoder.eval()

        with torch.no_grad():
            tokens_normal = tokenizer(normal_prompts).to(self.device)
            tokens_abnormal = tokenizer(abnormal_prompts).to(self.device)
            t_norm = F.normalize(text_encoder.encode_text(tokens_normal), p=2, dim=-1).detach()
            t_abn = F.normalize(text_encoder.encode_text(tokens_abnormal), p=2, dim=-1).detach()

        del text_encoder
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cls_names, t_norm.to(self.device), t_abn.to(self.device)

    def _select_text_priors(self, cls_names):
        if self.T_norm_prior is None or self.T_abn_prior is None:
            raise RuntimeError('Text priors are not initialized.')

        if self.T_norm_prior.shape[0] == 1:
            batch_size = len(cls_names)
            return self.T_norm_prior.expand(batch_size, -1), self.T_abn_prior

        class_ids = []
        for cls_name in cls_names:
            key = str(cls_name).lower()
            if key not in self.prior_name_to_idx:
                raise KeyError(
                    f'No cached BiomedCLIP prior found for class `{cls_name}`. '
                    f'Available classes: {sorted(self.prior_name_to_idx.keys())}.'
                )
            class_ids.append(self.prior_name_to_idx[key])
        class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.long)
        return self.T_norm_prior.index_select(0, class_ids), self.T_abn_prior

    def _class_names_to_indices(self, cls_names):
        class_ids = []
        for cls_name in cls_names:
            key = str(cls_name).lower()
            if key not in self.balance_class_to_idx:
                raise KeyError(
                    f'No balancing entry found for class `{cls_name}`. '
                    f'Available classes: {sorted(self.balance_class_to_idx.keys())}.'
                )
            class_ids.append(self.balance_class_to_idx[key])
        return torch.tensor(class_ids, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def _update_class_loss_ema(self, class_indices, sample_losses):
        if self.class_loss_ema.numel() == 0:
            return

        detached_losses = sample_losses.detach()
        class_sums = torch.zeros_like(self.class_loss_ema)
        class_counts = torch.zeros_like(self.class_loss_ema)
        class_sums.index_add_(0, class_indices, detached_losses)
        class_counts.index_add_(0, class_indices, torch.ones_like(detached_losses))

        valid_mask = class_counts > 0
        if valid_mask.any():
            class_means = torch.zeros_like(self.class_loss_ema)
            class_means[valid_mask] = class_sums[valid_mask] / class_counts[valid_mask]
            self.class_loss_ema[valid_mask] = (
                self.class_loss_ema_momentum * self.class_loss_ema[valid_mask]
                + (1.0 - self.class_loss_ema_momentum) * class_means[valid_mask]
            )

    def _get_class_weights(self, class_indices):
        if self.class_loss_ema.numel() == 0:
            return torch.ones_like(class_indices, dtype=torch.float32)

        global_mean = self.class_loss_ema.mean()
        weight_table = torch.ones_like(self.class_loss_ema)
        weight_table = torch.where(
            self.class_loss_ema > global_mean,
            torch.full_like(weight_table, self.class_weight_high),
            torch.where(
                self.class_loss_ema < global_mean,
                torch.full_like(weight_table, self.class_weight_low),
                weight_table,
            ),
        )
        return weight_table.index_select(0, class_indices)

    def _get_adaptive_mc_weight(self):
        if self.adaptive_mc_warmup_epochs <= 0:
            return self.adaptive_mc_weight_end
        progress = min(max(float(self.epoch) / float(self.adaptive_mc_warmup_epochs), 0.0), 1.0)
        return self.adaptive_mc_weight_start + (self.adaptive_mc_weight_end - self.adaptive_mc_weight_start) * progress

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly'].cuda().long().view(-1)
        self.bs = self.imgs.shape[0]

    def forward(self):
        self.t_norm_batch, self.t_abn_batch = None, None
        text_condition = None
        if self.use_adaptive_mc:
            self.t_norm_batch, self.t_abn_batch = self._select_text_priors(self.cls_name)
            text_condition = self.t_norm_batch
        self.feats_t, self.feats_s, self.f_global = self.net(
            self.imgs,
            self.cls_name,
            text_condition=text_condition,
            return_teacher_features=True,
        )

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)

            loss_adaptive_mc = loss_mse.new_tensor(0.0)
            adaptive_mc_weight = 0.0
            if self.use_adaptive_mc:
                batch_class_indices = self._class_names_to_indices(self.cls_name)
                adaptive_mc_output = self.loss_terms['adaptive_mc'](
                    self.f_global,
                    self.t_norm_batch,
                    self.t_abn_batch,
                    self.anomaly,
                    return_details=True,
                )
                self._update_class_loss_ema(batch_class_indices, adaptive_mc_output['selected_losses'])
                class_weights = self._get_class_weights(batch_class_indices)
                loss_adaptive_mc = torch.sum(adaptive_mc_output['selected_losses'] * class_weights) / class_weights.sum().clamp_min(1e-6)
                adaptive_mc_weight = self._get_adaptive_mc_weight()

            loss_l1 = loss_mse.new_tensor(0.0)
            for module in self.net.modules():
                if hasattr(module, 'l1_penalty'):
                    loss_l1 = loss_l1 + module.l1_penalty

            total_loss = loss_mse + self.lambda_l1 * loss_l1 + adaptive_mc_weight * loss_adaptive_mc

        self.backward_term(total_loss, self.optim)

        update_log_term(
            self.log_terms.get('pixel'),
            reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
            1,
            self.master,
        )
        update_log_term(
            self.log_terms.get('adaptive_mc'),
            reduce_tensor(loss_adaptive_mc, self.world_size).clone().detach().item(),
            1,
            self.master,
        )
        update_log_term(
            self.log_terms.get('total'),
            reduce_tensor(total_loss, self.world_size).clone().detach().item(),
            1,
            self.master,
        )

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)
        self.reset(isTrain=False)
        imgs_masks, anomaly_maps, cls_names, anomalys, sample_anomalys, sample_predicts = [], [], [], [], [], []
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
            update_log_term(
                self.log_terms.get('pixel'),
                reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                1,
                self.master,
            )
            anomaly_map, _ = self.evaluator.cal_anomaly_map(
                self.feats_t,
                self.feats_s,
                [self.imgs.shape[2], self.imgs.shape[3]],
                uni_am=False,
                amap_mode='add',
                gaussian_sigma=4,
            )
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map)
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))
            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix='Test'), self.master, None)
                    log_msg(self.logger, msg)

        if self.cfg.dist:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
                valid_results = False
                while not valid_results:
                    results_files = glob.glob(f'{self.tmp_dir}/*.pth')
                    if len(results_files) != self.cfg.world_size:
                        time.sleep(1)
                    else:
                        idx_result = 0
                        while idx_result < self.cfg.world_size:
                            results_file = results_files[idx_result]
                            try:
                                result = torch.load(results_file)
                                for k, v in result.items():
                                    results[k].extend(v)
                                idx_result += 1
                            except Exception:
                                time.sleep(1)
                        valid_results = True
        else:
            results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)

        if self.master:
            results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            msg = {}
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = len(self.cls_names) > 1 and idx == len(self.cls_names) - 1
                msg['Name'].append('Avg') if avg_act else None
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            msg = tabulate.tabulate(msg, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center', stralign='center')
            log_msg(self.logger, f'\n{msg}')
