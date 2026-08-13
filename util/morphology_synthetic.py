import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_batch_image(image):
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim != 4:
        raise ValueError(f'image must have shape [B,C,H,W] or [C,H,W], got {tuple(image.shape)}')
    return image, False


def _ensure_mask(mask, image, target_hw):
    if mask is None:
        return None
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask, device=image.device, dtype=image.dtype)
    else:
        mask = mask.to(device=image.device, dtype=image.dtype)
    if mask.ndim == 2:
        mask = mask.view(1, 1, *mask.shape)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 4:
        raise ValueError(f'mask must have shape [H,W], [B,H,W], or [B,1,H,W], got {tuple(mask.shape)}')
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    if mask.shape[0] != image.shape[0]:
        raise ValueError(f'mask batch size {mask.shape[0]} does not match image batch size {image.shape[0]}')
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != target_hw:
        mask = F.interpolate(mask, size=target_hw, mode='nearest')
    return mask > 0.5


def _odd_kernel_size(sigma):
    size = int(math.ceil(float(sigma) * 6.0))
    size = max(size, 3)
    return size + 1 if size % 2 == 0 else size


def _gaussian_kernel2d(sigma, device, dtype):
    sigma = max(float(sigma), 1e-3)
    size = _odd_kernel_size(sigma)
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel_1d = torch.exp(-(coords * coords) / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d.view(1, 1, size, size)


def gaussian_blur(mask, sigma):
    kernel = _gaussian_kernel2d(sigma, mask.device, mask.dtype)
    padding = kernel.shape[-1] // 2
    return F.conv2d(mask, kernel, padding=padding)


def _erode(mask, iters):
    out = mask.float()
    for _ in range(max(int(iters), 0)):
        out = 1.0 - F.max_pool2d(1.0 - out, kernel_size=3, stride=1, padding=1)
    return out


def _dilate(mask, iters):
    out = mask.float()
    for _ in range(max(int(iters), 0)):
        out = F.max_pool2d(out, kernel_size=3, stride=1, padding=1)
    return out


def _mask_bbox_stats(mask_2d):
    ys, xs = torch.where(mask_2d > 0.5)
    if ys.numel() == 0:
        return None
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    h = (y1 - y0).to(dtype=torch.float32).clamp_min(1.0)
    w = (x1 - x0).to(dtype=torch.float32).clamp_min(1.0)
    return int(y0), int(y1), int(x0), int(x1), float(w / h)


def _component_count(mask):
    if mask.ndim == 4:
        return [_component_count(mask[i, 0]) for i in range(mask.shape[0])]
    binary = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    for y in range(height):
        for x in range(width):
            if binary[y, x] == 0 or visited[y, x]:
                continue
            count += 1
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return count


def _perimeter(mask):
    x = mask.float()
    up = F.pad(x[..., 1:, :], (0, 0, 0, 1))
    down = F.pad(x[..., :-1, :], (0, 0, 1, 0))
    left = F.pad(x[..., :, 1:], (0, 1, 0, 0))
    right = F.pad(x[..., :, :-1], (1, 0, 0, 0))
    edge = (x > 0.5) & ((up < 0.5) | (down < 0.5) | (left < 0.5) | (right < 0.5))
    return edge.float().sum(dim=(-2, -1))


def mask_shape_stats(mask):
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ValueError(f'mask must have shape [B,1,H,W] or [B,H,W], got {tuple(mask.shape)}')
    mask = mask[:, :1].float()
    batch_size, _, height, width = mask.shape
    area = mask.sum(dim=(-2, -1)).clamp_min(0.0)
    area_ratio = area / float(height * width)
    per = _perimeter(mask).clamp_min(1.0)
    compactness = (per * per) / (4.0 * math.pi * area.clamp_min(1.0))
    out = {
        'area_ratio': area_ratio.detach(),
        'compactness': compactness.detach(),
        'perimeter_area_ratio': (per / area.clamp_min(1.0)).detach(),
        'component_count': torch.tensor(_component_count(mask), device=mask.device, dtype=mask.dtype),
    }
    aspect = []
    eccentricity = []
    for idx in range(batch_size):
        m = mask[idx, 0] > 0.5
        bbox = _mask_bbox_stats(m)
        if bbox is None:
            aspect.append(1.0)
            eccentricity.append(0.0)
            continue
        aspect.append(float(bbox[-1]))
        ys, xs = torch.where(m)
        coords = torch.stack([ys.float(), xs.float()], dim=1)
        coords = coords - coords.mean(dim=0, keepdim=True)
        if coords.shape[0] < 3:
            eccentricity.append(0.0)
        else:
            cov = coords.t().matmul(coords) / coords.shape[0]
            eigvals = torch.linalg.eigvalsh(cov).clamp_min(1e-6)
            eccentricity.append(float(torch.sqrt(1.0 - eigvals[0] / eigvals[1]).clamp(0.0, 1.0)))
    out['aspect_ratio'] = torch.tensor(aspect, device=mask.device, dtype=mask.dtype)
    out['eccentricity'] = torch.tensor(eccentricity, device=mask.device, dtype=mask.dtype)
    out['boundary_irregularity'] = torch.sqrt(out['compactness'].clamp_min(0.0))
    return out


def _stats_to_python(stats):
    result = {}
    for key, value in stats.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu().tolist()
        else:
            result[key] = value
    return result


def _quantile_sample(values, batch_size, device, dtype):
    values = torch.as_tensor(values, device=device, dtype=dtype).flatten()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    indices = torch.randint(values.numel(), (batch_size,), device=device)
    return values.index_select(0, indices)


class MorphologyAwareMaskGenerator(nn.Module):
    """Generate synthetic lesion masks from public-dataset morphology priors."""

    def __init__(
        self,
        use_morphology_prior=False,
        morphology_prior_path=None,
        area_ratio_range=(0.005, 0.08),
        aspect_ratio_tolerance=0.5,
        max_mask_retry=8,
        elastic_deform_prob=0.25,
        multi_component_prob=0.2,
        blur_sigma_range=(3.0, 7.0),
        threshold_quantile_range=(0.75, 0.93),
        foreground_threshold=5.0 / 255.0,
        foreground_erode_iters=0,
        boundary_tolerance=1.0,
        verbose_stats=False,
    ):
        super().__init__()
        self.use_morphology_prior = bool(use_morphology_prior)
        self.morphology_prior_path = morphology_prior_path
        self.area_ratio_range = tuple(area_ratio_range)
        self.aspect_ratio_tolerance = float(aspect_ratio_tolerance)
        self.max_mask_retry = int(max_mask_retry)
        self.elastic_deform_prob = float(elastic_deform_prob)
        self.multi_component_prob = float(multi_component_prob)
        self.blur_sigma_range = tuple(blur_sigma_range)
        self.threshold_quantile_range = tuple(threshold_quantile_range)
        self.foreground_threshold = float(foreground_threshold)
        self.foreground_erode_iters = int(foreground_erode_iters)
        self.boundary_tolerance = float(boundary_tolerance)
        self.verbose_stats = bool(verbose_stats)
        self.prior = self._load_prior(morphology_prior_path) if self.use_morphology_prior else None
        self.last_target_stats = None
        self.last_initial_stats = None
        self.last_final_stats = None
        self.last_stat_errors = None

    def _load_prior(self, path):
        if path is None:
            return None
        path = Path(path)
        if not path.exists():
            return None
        if path.suffix.lower() == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        if path.suffix.lower() == '.npz':
            data = np.load(path, allow_pickle=False)
            return {key: data[key].tolist() for key in data.files}
        raise ValueError(f'Unsupported morphology prior format: {path}')

    def _resolve_prior(self, morphology_prior):
        if morphology_prior is not None:
            return morphology_prior
        return self.prior if self.use_morphology_prior else None

    def _sample_targets(self, batch_size, image, morphology_prior):
        device, dtype = image.device, image.dtype
        lo, hi = self.area_ratio_range
        targets = {
            'area_ratio': lo + (hi - lo) * torch.rand(batch_size, device=device, dtype=dtype),
            'aspect_ratio': torch.ones(batch_size, device=device, dtype=dtype),
            'eccentricity': torch.zeros(batch_size, device=device, dtype=dtype),
            'component_count': torch.ones(batch_size, device=device, dtype=dtype),
            'boundary_irregularity': torch.ones(batch_size, device=device, dtype=dtype),
        }
        if not morphology_prior:
            return targets
        samples = morphology_prior.get('samples', morphology_prior) if isinstance(morphology_prior, dict) else {}
        if isinstance(samples, list):
            columns = {}
            for key in targets:
                values = [float(item[key]) for item in samples if key in item and np.isfinite(float(item[key]))]
                if values:
                    columns[key] = values
        elif isinstance(samples, dict):
            columns = samples
        else:
            columns = {}
        for key in targets:
            sampled = _quantile_sample(columns.get(key, []), batch_size, device, dtype)
            if sampled is not None:
                targets[key] = sampled
        targets['area_ratio'] = targets['area_ratio'].clamp(min=max(lo, 1e-6), max=max(hi, lo + 1e-6))
        targets['aspect_ratio'] = targets['aspect_ratio'].clamp(0.15, 8.0)
        targets['component_count'] = targets['component_count'].round().clamp_min(1.0)
        return targets

    def _foreground_from_image(self, image):
        foreground = image.detach().float()
        if foreground.min() < -0.1 or foreground.max() > 1.1:
            channels = min(image.shape[1], 3)
            mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)[:, :channels]
            std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)[:, :channels]
            foreground = image[:, :channels] * std + mean
        foreground = foreground.max(dim=1, keepdim=True).values > self.foreground_threshold
        if self.foreground_erode_iters > 0:
            foreground = _erode(foreground.float(), self.foreground_erode_iters) > 0.5
        return foreground

    def _base_mask(self, image, allowed_region, targets):
        batch_size, _, height, width = image.shape
        device, dtype = image.device, image.dtype
        masks = []
        sigmas = self.blur_sigma_range[0] + (
            self.blur_sigma_range[1] - self.blur_sigma_range[0]
        ) * torch.rand(batch_size, device=device, dtype=dtype)
        qlo, qhi = self.threshold_quantile_range
        quantiles = qlo + (qhi - qlo) * torch.rand(batch_size, device=device, dtype=dtype)
        for idx in range(batch_size):
            requested_components = int(targets['component_count'][idx].round().clamp(1, 5).detach().cpu())
            num_fields = requested_components if requested_components > 1 and torch.rand((), device=device) < self.multi_component_prob else 1
            mask = torch.zeros((1, 1, height, width), device=device, dtype=dtype)
            for _ in range(num_fields):
                noise = torch.randn((1, 1, height, width), device=device, dtype=dtype)
                smooth = gaussian_blur(noise, float(sigmas[idx]))
                threshold = torch.quantile(smooth.flatten(), float(quantiles[idx]))
                mask = torch.maximum(mask, (smooth > threshold).float())
            mask = mask * allowed_region[idx:idx + 1].float()
            masks.append(mask)
        return torch.cat(masks, dim=0)

    def _anisotropic_scale(self, sample, target_aspect):
        bbox = _mask_bbox_stats(sample[0] > 0.5)
        if bbox is None:
            return sample
        y0, y1, x0, x1, current_aspect = bbox
        crop = sample[:, y0:y1, x0:x1].unsqueeze(0)
        scale = math.sqrt(max(float(target_aspect), 1e-6) / max(current_aspect, 1e-6))
        new_w = max(1, int(round(crop.shape[-1] * scale)))
        new_h = max(1, int(round(crop.shape[-2] / scale)))
        resized = F.interpolate(crop, size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
        out = torch.zeros_like(sample)
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        oy0 = max(0, cy - new_h // 2)
        ox0 = max(0, cx - new_w // 2)
        oy1 = min(out.shape[-2], oy0 + new_h)
        ox1 = min(out.shape[-1], ox0 + new_w)
        out[:, oy0:oy1, ox0:ox1] = resized[:, :oy1 - oy0, :ox1 - ox0]
        return out.clamp(0.0, 1.0)

    def _resize_to_area(self, sample, target_area_ratio):
        _, height, width = sample.shape
        target_area = max(float(target_area_ratio) * height * width, 1.0)
        current_area = float((sample > 0.5).sum().detach().cpu())
        if current_area <= 0:
            return sample
        scale = math.sqrt(target_area / current_area)
        bbox = _mask_bbox_stats(sample[0] > 0.5)
        if bbox is None:
            return sample
        y0, y1, x0, x1, _ = bbox
        crop = sample[:, y0:y1, x0:x1].unsqueeze(0)
        new_h = max(1, int(round(crop.shape[-2] * scale)))
        new_w = max(1, int(round(crop.shape[-1] * scale)))
        resized = F.interpolate(crop, size=(new_h, new_w), mode='bilinear', align_corners=False)[0]
        out = torch.zeros_like(sample)
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        oy0 = max(0, cy - new_h // 2)
        ox0 = max(0, cx - new_w // 2)
        oy1 = min(height, oy0 + new_h)
        ox1 = min(width, ox0 + new_w)
        out[:, oy0:oy1, ox0:ox1] = resized[:, :oy1 - oy0, :ox1 - ox0]
        return out

    def _elastic_deform(self, sample):
        if torch.rand((), device=sample.device) > self.elastic_deform_prob:
            return sample
        _, height, width = sample.shape
        dtype, device = sample.dtype, sample.device
        disp = torch.randn((1, 2, max(2, height // 16), max(2, width // 16)), device=device, dtype=dtype)
        disp = F.interpolate(disp, size=(height, width), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
        warped = F.grid_sample(sample.unsqueeze(0), grid + 0.04 * disp, mode='bilinear', padding_mode='zeros', align_corners=False)
        return warped[0]

    def _adjust_mask(self, mask, allowed_region, targets):
        adjusted = []
        for idx in range(mask.shape[0]):
            sample = mask[idx]
            sample = self._resize_to_area(sample, targets['area_ratio'][idx])
            sample = self._anisotropic_scale(sample, targets['aspect_ratio'][idx])
            sample = self._elastic_deform(sample).unsqueeze(0)
            target_boundary = float(targets['boundary_irregularity'][idx].detach().cpu())
            current = mask_shape_stats((sample > 0.5).float())['boundary_irregularity'][0]
            if float(current.detach().cpu()) > target_boundary + self.boundary_tolerance:
                sample = _erode(_dilate(sample, 1), 1)
            elif float(current.detach().cpu()) < max(target_boundary - self.boundary_tolerance, 1.0):
                sample = _dilate(_erode(sample, 1), 1)
            sample = sample * allowed_region[idx:idx + 1].float()
            adjusted.append((sample > 0.5).float())
        return torch.cat(adjusted, dim=0)

    def _errors(self, stats, targets):
        return {
            'area_ratio_abs': (stats['area_ratio'] - targets['area_ratio']).abs(),
            'aspect_ratio_abs': (stats['aspect_ratio'] - targets['aspect_ratio']).abs(),
            'eccentricity_abs': (stats['eccentricity'] - targets['eccentricity']).abs(),
            'component_count_abs': (stats['component_count'] - targets['component_count']).abs(),
            'boundary_irregularity_abs': (stats['boundary_irregularity'] - targets['boundary_irregularity']).abs(),
        }

    def _within_tolerance(self, stats, targets):
        area_error = (stats['area_ratio'] - targets['area_ratio']).abs()
        aspect_error = (stats['aspect_ratio'] - targets['aspect_ratio']).abs()
        area_tol = torch.maximum(targets['area_ratio'] * 0.75, targets['area_ratio'].new_tensor(0.002))
        valid_area = area_error <= area_tol
        valid_aspect = aspect_error <= self.aspect_ratio_tolerance
        nonempty = stats['area_ratio'] > 0
        return bool((valid_area & valid_aspect & nonempty).all().detach().cpu())

    def forward(self, image, anatomy_mask=None, morphology_prior=None):
        image, squeezed = _as_batch_image(image)
        batch_size, _, height, width = image.shape
        prior = self._resolve_prior(morphology_prior)
        anatomy = _ensure_mask(anatomy_mask, image, (height, width))
        allowed_region = anatomy if anatomy is not None else self._foreground_from_image(image)
        if allowed_region.float().sum() <= 0:
            allowed_region = torch.ones((batch_size, 1, height, width), device=image.device, dtype=torch.bool)
        targets = self._sample_targets(batch_size, image, prior)
        best_mask = None
        best_stats = None
        best_error = None
        initial_stats = None
        retries = self.max_mask_retry if prior else 1
        for attempt in range(max(retries, 1)):
            base = self._base_mask(image, allowed_region, targets)
            if attempt == 0:
                initial_stats = mask_shape_stats(base)
            candidate = self._adjust_mask(base, allowed_region, targets) if prior else base
            stats = mask_shape_stats(candidate)
            errors = self._errors(stats, targets)
            score = errors['area_ratio_abs'].mean() + 0.05 * errors['aspect_ratio_abs'].mean()
            if best_error is None or float(score.detach().cpu()) < best_error:
                best_mask = candidate
                best_stats = stats
                best_error = float(score.detach().cpu())
            if prior and self._within_tolerance(stats, targets):
                best_mask = candidate
                best_stats = stats
                break
        self.last_target_stats = _stats_to_python(targets)
        self.last_initial_stats = _stats_to_python(initial_stats if initial_stats is not None else mask_shape_stats(best_mask))
        self.last_final_stats = _stats_to_python(best_stats)
        self.last_stat_errors = _stats_to_python(self._errors(best_stats, targets))
        if self.verbose_stats:
            print(json.dumps(self.stats_report(), indent=2))
        return best_mask[0] if squeezed else best_mask

    def stats_report(self):
        return {
            'target': self.last_target_stats,
            'initial': self.last_initial_stats,
            'final': self.last_final_stats,
            'error': self.last_stat_errors,
        }
