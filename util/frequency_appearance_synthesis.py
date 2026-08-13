from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from util.morphology_synthetic import MorphologyAwareMaskGenerator, gaussian_blur


def _as_batch_image(image):
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim != 4:
        raise ValueError(f'normal_image must have shape [B,C,H,W] or [C,H,W], got {tuple(image.shape)}')
    return image, False


def _ensure_mask(mask, image):
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
        raise ValueError(f'synthetic_mask must have shape [H,W], [B,H,W], or [B,1,H,W], got {tuple(mask.shape)}')
    if mask.shape[0] == 1 and image.shape[0] > 1:
        mask = mask.expand(image.shape[0], -1, -1, -1)
    if mask.shape[0] != image.shape[0]:
        raise ValueError(f'mask batch size {mask.shape[0]} does not match image batch size {image.shape[0]}')
    if mask.shape[1] != 1:
        mask = mask[:, :1]
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:], mode='nearest')
    return mask.clamp(0.0, 1.0)


def _normalize_field(field):
    flat = field.flatten(2)
    mean = flat.mean(dim=-1).view(field.shape[0], field.shape[1], 1, 1)
    std = flat.std(dim=-1).view(field.shape[0], field.shape[1], 1, 1).clamp_min(1e-6)
    return (field - mean) / std


def _blur_channels(image, sigma):
    channels = [gaussian_blur(image[:, idx:idx + 1], sigma) for idx in range(image.shape[1])]
    return torch.cat(channels, dim=1)


def _soft_mask(mask, sigma):
    soft = gaussian_blur(mask.float(), sigma).clamp(0.0, 1.0)
    return soft * (mask > 0.0).to(dtype=soft.dtype)


def _to_uint8_image(tensor):
    image = tensor.detach().cpu().float()
    if image.ndim == 4:
        image = image[0]
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    elif image.shape[0] > 3:
        image = image[:3]
    lo = torch.quantile(image.flatten(), 0.01)
    hi = torch.quantile(image.flatten(), 0.99)
    image = ((image - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)
    return (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


class FrequencyAwareLesionAppearanceSynthesis(nn.Module):
    """Synthesize frequency-aware lesion appearance under one provided mask."""

    def __init__(
        self,
        mask_generator=None,
        mode='random_band',
        boundary_sigma=2.0,
        low_sigma=11.0,
        mid_sigma_small=2.0,
        mid_sigma_large=6.0,
        high_sigma=1.2,
        low_strength=0.20,
        mid_strength=0.14,
        high_strength=0.08,
        joint_weights=(0.45, 0.35, 0.20),
        clamp_output=False,
        output_range=(0.0, 1.0),
    ):
        super().__init__()
        self.mask_generator = mask_generator if mask_generator is not None else MorphologyAwareMaskGenerator()
        self.mode = mode
        self.boundary_sigma = float(boundary_sigma)
        self.low_sigma = float(low_sigma)
        self.mid_sigma_small = float(mid_sigma_small)
        self.mid_sigma_large = float(mid_sigma_large)
        self.high_sigma = float(high_sigma)
        self.low_strength = float(low_strength)
        self.mid_strength = float(mid_strength)
        self.high_strength = float(high_strength)
        self.joint_weights = tuple(float(x) for x in joint_weights)
        self.clamp_output = bool(clamp_output)
        self.output_range = tuple(output_range)

    def _low_frequency(self, normal_image, soft_mask):
        batch_size, channels, height, width = normal_image.shape
        field = torch.randn((batch_size, 1, height, width), device=normal_image.device, dtype=normal_image.dtype)
        field = _normalize_field(gaussian_blur(field, self.low_sigma)).expand(-1, channels, -1, -1)
        large_structure = _blur_channels(normal_image, self.low_sigma) - normal_image
        channel_scale = 0.75 + 0.5 * torch.rand((batch_size, channels, 1, 1), device=normal_image.device, dtype=normal_image.dtype)
        perturbation = self.low_strength * channel_scale * field + 0.5 * large_structure
        return perturbation * soft_mask

    def _mid_frequency(self, normal_image, soft_mask):
        batch_size, channels, height, width = normal_image.shape
        noise = torch.randn((batch_size, channels, height, width), device=normal_image.device, dtype=normal_image.dtype)
        local_pattern = gaussian_blur(noise, self.mid_sigma_small) - gaussian_blur(noise, self.mid_sigma_large)
        local_pattern = _normalize_field(local_pattern)
        tissue = _blur_channels(normal_image, self.mid_sigma_small) - _blur_channels(normal_image, self.mid_sigma_large)
        perturbation = self.mid_strength * local_pattern + 0.6 * tissue
        return perturbation * soft_mask

    def _high_frequency(self, normal_image, soft_mask):
        batch_size, channels, height, width = normal_image.shape
        noise = torch.randn((batch_size, channels, height, width), device=normal_image.device, dtype=normal_image.dtype)
        fine = _normalize_field(noise - gaussian_blur(noise, self.high_sigma))
        image_edge = normal_image - _blur_channels(normal_image, self.high_sigma)
        perturbation = self.high_strength * fine + 0.8 * image_edge
        return perturbation * soft_mask

    def _select_appearance(self, low, mid, high, mode):
        if mode in ('low', 'low_frequency'):
            return low, 'low'
        if mode in ('mid', 'mid_frequency'):
            return mid, 'mid'
        if mode in ('high', 'high_frequency'):
            return high, 'high'
        if mode in ('joint', 'joint_fusion', 'fused', 'multi_band'):
            weights = low.new_tensor(self.joint_weights).view(3, 1, 1, 1, 1)
            weights = weights / weights.sum().clamp_min(1e-6)
            return (torch.stack([low, mid, high], dim=0) * weights).sum(dim=0), 'joint_fusion'
        if mode not in ('random', 'random_band', 'single'):
            raise ValueError(f'Unsupported frequency synthesis mode: {mode}')
        band_idx = torch.randint(0, 3, (low.shape[0],), device=low.device)
        stacked = torch.stack([low, mid, high], dim=1)
        gather_idx = band_idx.view(low.shape[0], 1, 1, 1, 1).expand(-1, 1, *low.shape[1:])
        selected = stacked.gather(dim=1, index=gather_idx).squeeze(1)
        names = ('low', 'mid', 'high')
        return selected, [names[int(idx)] for idx in band_idx.detach().cpu()]

    def forward(
        self,
        normal_image,
        synthetic_mask=None,
        anatomy_mask=None,
        morphology_prior=None,
        mode=None,
        return_bands=True,
    ):
        normal_image, squeezed = _as_batch_image(normal_image)
        if synthetic_mask is None:
            synthetic_mask = self.mask_generator(normal_image, anatomy_mask=anatomy_mask, morphology_prior=morphology_prior)
        synthetic_mask = _ensure_mask(synthetic_mask, normal_image)
        soft_mask = _soft_mask(synthetic_mask, self.boundary_sigma).to(dtype=normal_image.dtype)

        low = self._low_frequency(normal_image, soft_mask)
        mid = self._mid_frequency(normal_image, soft_mask)
        high = self._high_frequency(normal_image, soft_mask)
        synthetic_appearance, selected_band = self._select_appearance(low, mid, high, self.mode if mode is None else mode)
        synthetic_image = normal_image * (1.0 - soft_mask) - synthetic_appearance * soft_mask
        if self.clamp_output:
            synthetic_image = synthetic_image.clamp(float(self.output_range[0]), float(self.output_range[1]))

        result = {
            'synthetic_appearance': synthetic_appearance,
            'synthetic_image': synthetic_image,
            'synthetic_mask': synthetic_mask,
            'soft_mask': soft_mask,
            'selected_band': selected_band,
        }
        if return_bands:
            low_result = normal_image * (1.0 - soft_mask) - low * soft_mask
            mid_result = normal_image * (1.0 - soft_mask) - mid * soft_mask
            high_result = normal_image * (1.0 - soft_mask) - high * soft_mask
            if self.clamp_output:
                lo, hi = float(self.output_range[0]), float(self.output_range[1])
                low_result = low_result.clamp(lo, hi)
                mid_result = mid_result.clamp(lo, hi)
                high_result = high_result.clamp(lo, hi)
            result.update(
                low_frequency_result=low_result,
                mid_frequency_result=mid_result,
                high_frequency_result=high_result,
                fused_appearance=synthetic_appearance,
            )
        if squeezed:
            for key, value in list(result.items()):
                if torch.is_tensor(value) and value.ndim == 4:
                    result[key] = value[0]
        return result


def save_frequency_synthesis_visualization(outputs, output_dir, prefix='sample', max_items=8):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        outputs['low_frequency_result'],
        outputs['mid_frequency_result'],
        outputs['high_frequency_result'],
        outputs['fused_appearance'],
        outputs['synthetic_image'],
    ]
    first_tensor = panels[0]
    batch_size = first_tensor.shape[0] if torch.is_tensor(first_tensor) and first_tensor.ndim == 4 else 1
    for idx in range(min(batch_size, max_items)):
        images = []
        for tensor in panels:
            item = tensor[idx:idx + 1] if tensor.ndim == 4 else tensor
            images.append(_to_uint8_image(item))
        height, width = images[0].shape[:2]
        canvas = Image.new('RGB', (width * len(images), height))
        for panel_idx, image in enumerate(images):
            canvas.paste(Image.fromarray(image), (panel_idx * width, 0))
        canvas.save(output_dir / f'{prefix}_{idx:03d}_frequency_synthesis.png')
