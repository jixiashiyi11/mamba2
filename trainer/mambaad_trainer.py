import copy
import os
import glob
import shutil
import time
import math

import numpy as np
import tabulate
import torch
import torch.nn.functional as F
import matplotlib.cm as cm
from PIL import Image

from util.util import able, log_msg, update_log_term
from util.net import get_timepc, reduce_tensor
from optim.scheduler import get_scheduler
from util.debug_eval import DebugEvalHelper, compute_foreground_masks_from_images

from ._base_trainer import BaseTrainer
from . import TRAINER


def _emit_metric_summary_table(trainer, results):
    msg = {}
    for idx, cls_name in enumerate(trainer.cls_names):
        metric_results = trainer.evaluator.run(results, cls_name, trainer.logger)
        msg['Name'] = msg.get('Name', [])
        msg['Name'].append(cls_name)
        avg_act = len(trainer.cls_names) > 1 and idx == len(trainer.cls_names) - 1
        msg['Name'].append('Avg') if avg_act else None
        for metric in trainer.metrics:
            metric_result = metric_results[metric] * 100
            trainer.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
            max_metric = max(trainer.metric_recorder[f'{metric}_{cls_name}'])
            max_metric_idx = trainer.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
            msg[metric] = msg.get(metric, [])
            msg[metric].append(metric_result)
            msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
            msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
            if avg_act:
                metric_result_avg = sum(msg[metric]) / len(msg[metric])
                trainer.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                max_metric = max(trainer.metric_recorder[f'{metric}_Avg'])
                max_metric_idx = trainer.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                msg[metric].append(metric_result_avg)
                msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
    table = tabulate.tabulate(msg, headers='keys', tablefmt='pipe', floatfmt='.3f', numalign='center', stralign='center')
    print(f'\n{table}')
    log_msg(trainer.logger, f'\n{table}')
    return table


def _write_debug_eval_safely(debug_helper, results, evaluator, logger):
    try:
        debug_helper.write_and_summarize(results, evaluator)
    except Exception as exc:
        msg = f'Warning: DebugEval failed after final metric table was printed: {exc}'
        print(msg)
        log_msg(logger, msg)


@TRAINER.register_module
class MAMBAADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(MAMBAADTrainer, self).__init__(cfg)
        self.device = torch.device(f'cuda:{cfg.local_rank}')
        self.lambda_l1 = getattr(cfg.loss, 'lambda_l1', 0.005)
        self.adaptive_mc_weight_start = getattr(cfg.loss, 'adaptive_mc_weight_start', 0.01)
        self.adaptive_mc_weight_end = getattr(cfg.loss, 'adaptive_mc_weight_end', 1.0)
        self.adaptive_mc_warmup_epochs = getattr(cfg.loss, 'adaptive_mc_warmup_epochs', 0)
        self.use_adaptive_mc = 'adaptive_mc' in self.loss_terms
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
            return self.T_norm_prior.expand(batch_size, -1), self.T_abn_prior.expand(batch_size, -1)

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
        return self.T_norm_prior.index_select(0, class_ids), self.T_abn_prior.index_select(0, class_ids)

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
        self.feats_t, self.feats_s, self.f_global = self.net(
            self.imgs,
            self.cls_name,
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
                t_norm_batch, _ = self._select_text_priors(self.cls_name)
                f_global = F.normalize(self.f_global, p=2, dim=1)
                t_norm_batch = F.normalize(t_norm_batch.to(device=f_global.device, dtype=f_global.dtype), p=2, dim=1)
                loss_adaptive_mc = 1.0 - torch.sum(f_global * t_norm_batch, dim=1).mean()
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
        update_log_term(
            self.log_terms.get('normal_align'),
            reduce_tensor(self.loss_dict['normal_align'], self.world_size).clone().detach().item(),
            1,
            self.master,
        )
        update_log_term(
            self.log_terms.get('token_consistency'),
            reduce_tensor(self.loss_dict['token_consistency'], self.world_size).clone().detach().item(),
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
            _emit_metric_summary_table(self, results)


@TRAINER.register_module
class MAMBAADZeroShotTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(MAMBAADZeroShotTrainer, self).__init__(cfg)
        self.device = torch.device(f'cuda:{cfg.local_rank}')
        base_output_dir = cfg.logdir if cfg.logdir is not None else cfg.trainer.checkpoint
        self.hr_anomaly_map_dir = os.path.join(base_output_dir, 'show_test_hr_anomaly_maps')
        self.save_hr_anomaly_maps = getattr(cfg.trainer, 'save_hr_anomaly_maps', False)
        if self.save_hr_anomaly_maps:
            os.makedirs(self.hr_anomaly_map_dir, exist_ok=True)
            print("测试异常图保存路径：", self.hr_anomaly_map_dir)

        if hasattr(cfg, 'data_train') and hasattr(cfg, 'data_test'):
            self._rebuild_cross_domain_loaders(cfg)
        self.debug_helper = DebugEvalHelper(cfg, self.logger, rank=self.rank, master=self.master)
        self._configure_eval_adapter_audit()

    def _net_module(self):
        return self.net.module if hasattr(self.net, 'module') else self.net

    def _configure_eval_adapter_audit(self):
        mode = str(getattr(self.cfg, 'eval_adapter_mode', 'trained')).lower()
        seed = getattr(self.cfg, 'eval_random_adapter_seed', 123)
        module = self._net_module()
        if not hasattr(module, 'set_eval_adapter_mode'):
            return
        module.set_eval_adapter_mode(mode)
        if mode == 'random':
            module.reset_adapter_parameters(seed=seed)
        adapter_norm, adapter_params = module.adapter_param_norm()
        log_msg(
            self.logger,
            f'==> EvalAdapterAudit mode={mode} random_seed={seed} '
            f'adapter_param_l2={adapter_norm:.6f} adapter_params={adapter_params}'
        )

    def _get_adapter_debug_numpy(self, batch_size):
        module = self._net_module()
        debug = getattr(module, 'last_adapter_debug', {}) if module is not None else {}
        out = {}
        for key in [
            'adapter_feature_delta_l2',
            'adapter_feature_delta_abs',
            'adapter_raw_l2',
            'adapter_refined_l2',
        ]:
            value = debug.get(key)
            if value is None:
                out[key] = np.full((batch_size,), np.nan, dtype=np.float32)
            else:
                out[key] = value.detach().cpu().float().numpy().reshape(-1)
        return out

    def _rebuild_cross_domain_loaders(self, cfg):
        try:
            from data import get_loader
        except ImportError as exc:
            raise ImportError(
                'Cross-domain mode requires ADer\'s native `data.get_loader` entrypoint.'
            ) from exc

        log_msg(self.logger, '==> Cross-domain mode detected, rebuilding source-train / target-test loaders')

        cfg_train = copy.copy(cfg)
        cfg_train.data = copy.deepcopy(cfg.data_train)
        train_loader, _ = get_loader(cfg_train)

        cfg_test = copy.copy(cfg)
        cfg_test.data = copy.deepcopy(cfg.data_test)
        _, test_loader = get_loader(cfg_test)

        self.train_loader = train_loader
        self.test_loader = test_loader

        cfg.data.train_size = len(self.train_loader)
        cfg.data.test_size = len(self.test_loader)
        cfg.data.train_length = self.train_loader.dataset.length
        cfg.data.test_length = self.test_loader.dataset.length
        self.scheduler = get_scheduler(cfg, self.optim)
        self.iter_full = cfg.trainer.iter_full
        self.epoch_full = cfg.trainer.epoch_full

        self.cls_names = list(self.test_loader.dataset.cls_names)
        self._sync_metric_recorder(self.cls_names)

        log_msg(self.logger, f'==> Source-domain train classes: {list(self.train_loader.dataset.cls_names)}')
        log_msg(self.logger, f'==> Target-domain test classes: {self.cls_names}')

    def _sync_metric_recorder(self, cls_names):
        existing = getattr(self, 'metric_recorder', {}) or {}
        synced = {}

        for idx, cls_name in enumerate(cls_names):
            for metric in self.metrics:
                key = f'{metric}_{cls_name}'
                synced[key] = list(existing.get(key, []))
                if idx == len(cls_names) - 1 and len(cls_names) > 1:
                    avg_key = f'{metric}_Avg'
                    synced[avg_key] = list(existing.get(avg_key, []))

        self.metric_recorder = synced
        self.cfg.trainer.metric_recorder = synced

    def _get_debug_anomaly_map_before_resize_shape(self):
        net = self.net.module if hasattr(self.net, 'module') else self.net
        grid_size = getattr(net, 'grid_size', None)
        if grid_size is None:
            return None
        return int(grid_size), int(grid_size)

    def _normalize_batch_paths(self, paths, batch_size, fill_value=''):
        if paths is None:
            return [fill_value] * batch_size
        if isinstance(paths, (list, tuple)):
            paths = list(paths)
        else:
            paths = [paths]
        paths = [str(path) for path in paths]
        if len(paths) < batch_size:
            paths.extend([fill_value] * (batch_size - len(paths)))
        return paths[:batch_size]

    def _candidate_data_roots(self):
        roots = []
        if hasattr(self.cfg, 'data_test') and getattr(self.cfg.data_test, 'root', None):
            roots.append(self.cfg.data_test.root)
        if getattr(self.cfg.data, 'root', None):
            roots.append(self.cfg.data.root)
        return roots

    def _resolve_mask_path(self, mask_path):
        if not mask_path:
            return None
        candidates = [mask_path]
        for root in self._candidate_data_roots():
            candidates.append(os.path.join(root, mask_path))
        for candidate in candidates:
            candidate = str(candidate).replace('/', os.sep)
            if os.path.exists(candidate):
                return candidate
        return None

    def _raw_positive_pixels_from_paths(self, mask_paths, final_masks):
        final_masks_np = final_masks.detach().cpu().numpy()
        if final_masks_np.ndim == 4:
            final_masks_np = np.squeeze(final_masks_np, axis=1)
        positives = []
        for idx, mask_path in enumerate(mask_paths):
            resolved = self._resolve_mask_path(mask_path)
            if resolved is None:
                positives.append(int(final_masks_np[idx].sum()))
                continue
            try:
                with Image.open(resolved) as mask_img:
                    mask_arr = np.asarray(mask_img.convert('L'))
                positives.append(int((mask_arr > 0).sum()))
            except Exception:
                positives.append(int(final_masks_np[idx].sum()))
        return positives

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly'].cuda().long().view(-1)
        self.img_path = inputs.get('img_path') if isinstance(inputs, dict) else None
        self.mask_path = inputs.get('mask_path') if isinstance(inputs, dict) else None
        self.bs = self.imgs.shape[0]

    def _save_high_res_anomaly_maps(self, anomaly_map, batch_idx):
        anomaly_map_np = anomaly_map.detach().cpu().numpy()
        for item_idx, sample_map in enumerate(anomaly_map_np):
            sample_name = f'rank{self.rank}_batch{batch_idx:04d}_item{item_idx:02d}'
            if self.img_path is not None:
                raw_path = self.img_path[item_idx] if isinstance(self.img_path, (list, tuple)) else self.img_path
                basename = os.path.splitext(os.path.basename(str(raw_path)))[0]
                sample_name = f'rank{self.rank}_{basename}'

            np.save(os.path.join(self.hr_anomaly_map_dir, f'{sample_name}.npy'), sample_map.astype(np.float32))

            sample_min = float(sample_map.min())
            sample_max = float(sample_map.max())
            if sample_max > sample_min:
                sample_map_norm = (sample_map - sample_min) / (sample_max - sample_min)
            else:
                sample_map_norm = np.zeros_like(sample_map, dtype=np.float32)
            sample_map_vis = (cm.jet(sample_map_norm)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(sample_map_vis).save(
                os.path.join(self.hr_anomaly_map_dir, f'{sample_name}.png')
            )

    def forward(self):
        cls_name_for_model = self.cls_name
        force_cls_name = getattr(self.cfg, 'eval_force_cls_name', None)
        if not self.net.training and force_cls_name:
            if isinstance(self.cls_name, (list, tuple)):
                cls_name_for_model = [str(force_cls_name)] * len(self.cls_name)
            else:
                cls_name_for_model = str(force_cls_name)
        if self.net.training:
            self.loss_dict = self.net(self.imgs, cls_names=cls_name_for_model)
            synthetic_cfg = getattr(self.cfg, 'synthetic_local_anomaly', None)
            if synthetic_cfg is not None and bool(getattr(synthetic_cfg, 'enabled', False)):
                synth_imgs, synth_masks = self._make_synthetic_local_anomaly_batch(synthetic_cfg)
                synth_out = self.net(
                    synth_imgs,
                    cls_names=cls_name_for_model,
                    return_anomaly_map=True,
                    compute_label_free=False,
                )
                synth_losses = self._synthetic_local_anomaly_loss(
                    synth_out['anomaly_map'],
                    synth_masks,
                    synthetic_cfg,
                )
                self.loss_dict.update(synth_losses)
                self.loss_dict['total'] = self.loss_dict['total'] + synth_losses['loss_synthetic_local_weighted']
                self.loss_dict['loss_total'] = self.loss_dict['total']
            self.total_loss = self.loss_dict['total']
        else:
            self.anomaly_map, self.image_score = self.net(self.imgs, cls_names=cls_name_for_model)

    def _make_synthetic_local_anomaly_batch(self, synthetic_cfg):
        imgs = self.imgs
        batch_size, _, height, width = imgs.shape
        device = imgs.device
        dtype = imgs.dtype
        mean = imgs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = imgs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        imgs_01 = (imgs * std + mean).clamp(0.0, 1.0)
        fg_threshold = float(getattr(synthetic_cfg, 'foreground_threshold', 5.0 / 255.0))
        foreground = (imgs_01.max(dim=1, keepdim=True).values > fg_threshold)
        masks = torch.zeros((batch_size, 1, height, width), device=device, dtype=dtype)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing='ij',
        )
        min_area = float(getattr(synthetic_cfg, 'min_area', 0.005))
        max_area = float(getattr(synthetic_cfg, 'max_area', 0.08))
        prob = float(getattr(synthetic_cfg, 'prob', 1.0))
        for idx in range(batch_size):
            if torch.rand((), device=device) > prob:
                continue
            fg_idx = torch.nonzero(foreground[idx, 0], as_tuple=False)
            if fg_idx.numel() > 0:
                center = fg_idx[torch.randint(fg_idx.shape[0], (1,), device=device).item()]
                cy = center[0].to(dtype)
                cx = center[1].to(dtype)
            else:
                cy = torch.randint(height, (1,), device=device).to(dtype)[0]
                cx = torch.randint(width, (1,), device=device).to(dtype)[0]
            area_ratio = min_area + (max_area - min_area) * float(torch.rand((), device=device))
            radius_base = math.sqrt(max(area_ratio, 1e-6)) * min(height, width)
            ry = radius_base * (0.55 + 0.9 * float(torch.rand((), device=device)))
            rx = radius_base * (0.55 + 0.9 * float(torch.rand((), device=device)))
            ellipse = (((yy - cy) / max(ry, 1.0)) ** 2 + ((xx - cx) / max(rx, 1.0)) ** 2) <= 1.0
            blob = ellipse & foreground[idx, 0]
            if int(blob.sum().detach().cpu()) < 8:
                blob = ellipse
            masks[idx, 0] = blob.to(dtype)

        noise_std = float(getattr(synthetic_cfg, 'noise_std', 0.18))
        intensity_delta = float(getattr(synthetic_cfg, 'intensity_delta', 0.35))
        noise = torch.randn_like(imgs_01) * noise_std
        signs = torch.where(
            torch.rand((batch_size, 1, 1, 1), device=device, dtype=dtype) > 0.5,
            torch.ones((batch_size, 1, 1, 1), device=device, dtype=dtype),
            -torch.ones((batch_size, 1, 1, 1), device=device, dtype=dtype),
        )
        channel_scale = 0.5 + torch.rand((batch_size, 3, 1, 1), device=device, dtype=dtype)
        perturb = signs * intensity_delta * channel_scale + noise
        synth_01 = (imgs_01 + masks * perturb).clamp(0.0, 1.0)
        synth_imgs = (synth_01 - mean) / std
        return synth_imgs, masks

    def _synthetic_local_anomaly_loss(self, anomaly_map, synth_masks, synthetic_cfg):
        if anomaly_map.ndim == 3:
            anomaly_map = anomaly_map.unsqueeze(1)
        if synth_masks.shape[-2:] != anomaly_map.shape[-2:]:
            synth_masks = F.interpolate(synth_masks, size=anomaly_map.shape[-2:], mode='nearest')
        temperature = max(float(getattr(synthetic_cfg, 'score_temperature', 0.1)), 1e-6)
        logits = anomaly_map / temperature
        targets = synth_masks.to(dtype=logits.dtype)
        loss_bce = F.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        loss_dice = (1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)).mean()
        bce_weight = float(getattr(synthetic_cfg, 'bce_weight', 1.0))
        dice_weight = float(getattr(synthetic_cfg, 'dice_weight', 1.0))
        loss_weight = float(getattr(synthetic_cfg, 'loss_weight', 0.1))
        loss_synthetic = bce_weight * loss_bce + dice_weight * loss_dice
        return {
            'loss_synthetic_local': loss_synthetic,
            'loss_synthetic_local_weighted': loss_weight * loss_synthetic,
            'loss_synthetic_bce': loss_bce.detach(),
            'loss_synthetic_dice': loss_dice.detach(),
            'synthetic_mask_ratio': targets.detach().mean(),
        }

    def optimize_parameters(self):
        with self.amp_autocast():
            self.forward()
            total_loss = self.total_loss

        if not torch.isfinite(total_loss):
            raise FloatingPointError(f'Non-finite total loss detected: {float(total_loss.detach().cpu())}')

        self.backward_term(total_loss, self.optim)

        update_log_term(
            self.log_terms.get('total'),
            reduce_tensor(total_loss, self.world_size).clone().detach().item(),
            1,
            self.master,
        )

        if self.master and self.iter % self.cfg.logging.train_log_per == 0:
            debug_names = [
                'loss_total',
                'loss_normal_align',
                'loss_token_consistency',
                'loss_adaptive_margin',
                'loss_score_separation',
                'sim_normal_mean',
                'sim_abnormal_mean',
                'adaptive_margin_mean',
                'score_train_topk_mean',
                'loss_synthetic_local',
                'loss_synthetic_bce',
                'loss_synthetic_dice',
                'synthetic_mask_ratio',
            ]
            debug_vals = {
                name: float(self.loss_dict[name].detach().cpu())
                for name in debug_names
                if name in self.loss_dict
            }
            has_nan = any(math.isnan(val) or math.isinf(val) for val in debug_vals.values())
            mem_mb = torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            debug_msg = ' '.join([f'{name}={val:.6f}' for name, val in debug_vals.items()])
            log_msg(self.logger, f'==> LossDebug {debug_msg} has_nan={has_nan} max_mem_mb={mem_mb:.1f}')

    @torch.no_grad()
    def test(self):
        if self.master:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.makedirs(self.tmp_dir, exist_ok=True)

        self.reset(isTrain=False)
        force_cls_name = getattr(self.cfg, 'eval_force_cls_name', None)
        if self.master and force_cls_name:
            log_msg(self.logger, f'==> EvalForceClsName model_cls_name={force_cls_name} metric_cls_names=original')
        imgs_masks, anomaly_maps, image_scores, cls_names, anomalys = [], [], [], [], []
        foreground_masks = []
        adapter_debug_results = {
            'adapter_feature_delta_l2': [],
            'adapter_feature_delta_abs': [],
            'adapter_raw_l2': [],
            'adapter_refined_l2': [],
        }
        img_paths, mask_paths, raw_positive_pixels, model_input_shapes, anomaly_map_shapes, gt_mask_shapes = [], [], [], [], [], []
        collect_eval_aux = self.debug_helper.enabled or getattr(self.evaluator, 'skip_tiny_mask_for_pixel', False)
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)

        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            anomaly_map = self.anomaly_map
            if self.save_hr_anomaly_maps:
                self._save_high_res_anomaly_maps(anomaly_map, batch_idx)

            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            batch_size = self.imgs.shape[0]
            adapter_debug = self._get_adapter_debug_numpy(batch_size)
            for key, value in adapter_debug.items():
                adapter_debug_results[key].append(value)
            foreground_mask = None
            if self.debug_helper.enabled and self.debug_helper.foreground_enabled:
                foreground_mask = compute_foreground_masks_from_images(
                    self.imgs,
                    self.debug_helper.foreground_threshold,
                )
                foreground_masks.append(foreground_mask.astype(np.uint8))
            if collect_eval_aux:
                paths = self._normalize_batch_paths(
                    self.img_path,
                    batch_size,
                    fill_value='',
                )
                if not any(paths):
                    paths = [f'rank{self.rank}_batch{batch_idx}_item{i}' for i in range(batch_size)]
                masks_paths = self._normalize_batch_paths(self.mask_path, batch_size, fill_value='')
                img_paths.append(np.array(paths))
                mask_paths.append(np.array(masks_paths))
                raw_positive_pixels.append(np.array(self._raw_positive_pixels_from_paths(masks_paths, self.imgs_mask)))

            if self.debug_helper.enabled:
                self.debug_helper.add_vis_batch(
                    self.imgs,
                    self.imgs_mask,
                    anomaly_map,
                    self.image_score,
                    self.cls_name,
                    self.anomaly,
                    self.img_path,
                    foreground_mask,
                )
                map_before_resize_shape = self._get_debug_anomaly_map_before_resize_shape()
                if map_before_resize_shape is None:
                    map_before_resize_shape = (anomaly_map.shape[-2], anomaly_map.shape[-1])
                model_input_shapes.append(np.tile(np.array([[self.imgs.shape[2], self.imgs.shape[3]]]), (batch_size, 1)))
                anomaly_map_shapes.append(np.tile(np.array([map_before_resize_shape]), (batch_size, 1)))
                gt_mask_shapes.append(np.tile(np.array([[self.imgs_mask.shape[-2], self.imgs_mask.shape[-1]]]), (batch_size, 1)))
            imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
            anomaly_maps.append(anomaly_map.cpu().numpy())
            image_scores.append(self.image_score.cpu().numpy())
            cls_names.append(np.array(self.cls_name))
            anomalys.append(self.anomaly.cpu().numpy().astype(int))

            t2 = get_timepc()
            update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
            print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
            if self.master:
                if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
                    msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix='Test'), self.master, None)
                    log_msg(self.logger, msg)

        if self.debug_helper.enabled:
            self.debug_helper.save_visualizations()

        if self.cfg.dist:
            results = dict(
                imgs_masks=imgs_masks,
                anomaly_maps=anomaly_maps,
                image_scores=image_scores,
                cls_names=cls_names,
                anomalys=anomalys,
                **adapter_debug_results,
            )
            if self.debug_helper.enabled:
                results.update(
                    model_input_shapes=model_input_shapes,
                    anomaly_map_shapes=anomaly_map_shapes,
                    gt_mask_shapes=gt_mask_shapes,
                )
                if self.debug_helper.foreground_enabled:
                    results.update(foreground_masks=foreground_masks)
            if collect_eval_aux:
                results.update(img_paths=img_paths, mask_paths=mask_paths, raw_positive_pixels=raw_positive_pixels)
            torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
            if self.master:
                results = dict(imgs_masks=[], anomaly_maps=[], image_scores=[], cls_names=[], anomalys=[])
                results.update({key: [] for key in adapter_debug_results})
                if self.debug_helper.enabled:
                    results.update(model_input_shapes=[], anomaly_map_shapes=[], gt_mask_shapes=[])
                    if self.debug_helper.foreground_enabled:
                        results.update(foreground_masks=[])
                if collect_eval_aux:
                    results.update(img_paths=[], mask_paths=[], raw_positive_pixels=[])
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
            results = dict(
                imgs_masks=imgs_masks,
                anomaly_maps=anomaly_maps,
                image_scores=image_scores,
                cls_names=cls_names,
                anomalys=anomalys,
                **adapter_debug_results,
            )
            if self.debug_helper.enabled:
                results.update(
                    model_input_shapes=model_input_shapes,
                    anomaly_map_shapes=anomaly_map_shapes,
                    gt_mask_shapes=gt_mask_shapes,
                )
                if self.debug_helper.foreground_enabled:
                    results.update(foreground_masks=foreground_masks)
            if collect_eval_aux:
                results.update(img_paths=img_paths, mask_paths=mask_paths, raw_positive_pixels=raw_positive_pixels)

        if self.master:
            results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
            _emit_metric_summary_table(self, results)
            _write_debug_eval_safely(self.debug_helper, results, self.evaluator, self.logger)
