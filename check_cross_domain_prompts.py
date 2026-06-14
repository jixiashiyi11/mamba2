import argparse
import importlib

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sanity-check BiomedCLIP text semantics for normal / abnormal / class prompts.'
    )
    parser.add_argument(
        '--config',
        default='configs.mambaad.mambaad_cross_domain',
        help='Config module path, for example configs.mambaad.mambaad_cross_domain',
    )
    parser.add_argument(
        '--classes',
        default='',
        help='Comma-separated class list. Defaults to all available classes in the config.',
    )
    parser.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device used for BiomedCLIP encoding.',
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=3,
        help='Number of decimal places for printed cosine similarities.',
    )
    return parser.parse_args()


def load_cfg(config_path):
    module = importlib.import_module(config_path)
    return module.cfg()


def resolve_classes(cfg, classes_arg):
    prompt_normal = getattr(cfg, 'prompt_normal', None)
    prompt_abnormal = getattr(cfg, 'prompt_abnormal', None)
    if not isinstance(prompt_normal, dict) or not isinstance(prompt_abnormal, dict):
        raise ValueError('Config must define dict-style `prompt_normal` and `prompt_abnormal`.')

    available = sorted(set(prompt_normal.keys()) & set(prompt_abnormal.keys()))
    if not available:
        raise ValueError('No overlapping class keys found between `prompt_normal` and `prompt_abnormal`.')

    if not classes_arg:
        return available

    selected = [item.strip().lower() for item in classes_arg.split(',') if item.strip()]
    missing = [item for item in selected if item not in available]
    if missing:
        raise ValueError(f'Unknown classes {missing}. Available classes: {available}')
    return selected


def build_class_prompts(cfg, class_names):
    class_prompt_map = getattr(cfg, 'class_prompts', None)
    class_prompt_template = getattr(cfg, 'class_prompt_template', 'A medical image of {class_name}')

    if isinstance(class_prompt_map, dict) and class_prompt_map:
        prompts = {}
        for class_name in class_names:
            if class_name not in class_prompt_map:
                raise KeyError(
                    f'Class `{class_name}` missing from `class_prompts`. '
                    f'Available keys: {sorted(class_prompt_map.keys())}'
                )
            prompts[class_name] = class_prompt_map[class_name]
        return prompts

    return {
        class_name: class_prompt_template.format(class_name=class_name)
        for class_name in class_names
    }


def encode_prompts(model, tokenizer, prompts, device):
    with torch.no_grad():
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
        feats = F.normalize(feats, p=2, dim=-1)
    return feats


def cosine_matrix(x, y):
    return x @ y.t()


def format_matrix(title, row_names, col_names, matrix, precision):
    fmt = f'{{:.{precision}f}}'
    col_width = max(max(len(name) for name in row_names + col_names), precision + 4)

    lines = [title]
    header = ' '.ljust(col_width + 2) + ' '.join(name.rjust(col_width) for name in col_names)
    lines.append(header)
    for idx, row_name in enumerate(row_names):
        values = ' '.join(fmt.format(matrix[idx, j].item()).rjust(col_width) for j in range(matrix.shape[1]))
        lines.append(row_name.ljust(col_width + 2) + values)
    return '\n'.join(lines)


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    try:
        import open_clip
    except ImportError as exc:
        raise ImportError('This script requires `open_clip_torch` to be installed.') from exc

    class_names = resolve_classes(cfg, args.classes)
    prompt_normal = cfg.prompt_normal
    prompt_abnormal = cfg.prompt_abnormal
    class_prompts = build_class_prompts(cfg, class_names)
    model_name = getattr(
        cfg,
        'biomedclip_model_name',
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    )

    device = torch.device(args.device)
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device)
    model.eval()

    normal_texts = [prompt_normal[name] for name in class_names]
    abnormal_texts = [prompt_abnormal[name] for name in class_names]
    class_texts = [class_prompts[name] for name in class_names]

    normal_feats = encode_prompts(model, tokenizer, normal_texts, device)
    abnormal_feats = encode_prompts(model, tokenizer, abnormal_texts, device)
    class_feats = encode_prompts(model, tokenizer, class_texts, device)

    print(f'Config: {args.config}')
    print(f'Model: {model_name}')
    print(f'Device: {device}')
    print(f'Classes: {class_names}')
    print()

    print('Per-class cosine summary')
    print('class | cos(norm,abn) | cos(norm,class) | cos(abn,class) | class closer to')
    issues = []
    for idx, class_name in enumerate(class_names):
        cos_norm_abn = torch.dot(normal_feats[idx], abnormal_feats[idx]).item()
        cos_norm_cls = torch.dot(normal_feats[idx], class_feats[idx]).item()
        cos_abn_cls = torch.dot(abnormal_feats[idx], class_feats[idx]).item()
        closer_to = 'normal' if cos_norm_cls >= cos_abn_cls else 'abnormal'
        if closer_to == 'abnormal':
            issues.append(class_name)
        print(
            f'{class_name:<10} | {cos_norm_abn:>8.{args.precision}f} '
            f'| {cos_norm_cls:>8.{args.precision}f} | {cos_abn_cls:>8.{args.precision}f} | {closer_to}'
        )

    print()
    print(format_matrix('Class-vs-Class Cosine Matrix', class_names, class_names, cosine_matrix(class_feats, class_feats), args.precision))
    print()
    print(format_matrix('Class-vs-Normal Cosine Matrix', class_names, class_names, cosine_matrix(class_feats, normal_feats), args.precision))
    print()
    print(format_matrix('Class-vs-Abnormal Cosine Matrix', class_names, class_names, cosine_matrix(class_feats, abnormal_feats), args.precision))
    print()

    if issues:
        print(f'Warning: class embeddings are closer to abnormal prompts for: {issues}')
    else:
        print('No immediate collapse sign: every class embedding is at least as close to its normal prompt as to its abnormal prompt.')


if __name__ == '__main__':
    main()
