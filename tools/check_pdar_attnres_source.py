import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text()


def _class_source(source, class_name):
    tree = ast.parse(source)
    node = next(
        item for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    return ast.get_source_segment(source, node)


def main():
    files = [
        "model/mambaad.py",
        "model/modules/adapters.py",
        "model/clip_ad.py",
        "trainer/clip_ad_trainer.py",
        "configs/clip_ad/clip_ad_supervised_mask_pdar_cssd.py",
        "configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa.py",
        "tools/test_pdar_attnres.py",
    ]
    for path in files:
        ast.parse(_read(path), filename=path)
    print("AST syntax: OK")

    mambaad = _read("model/mambaad.py")
    pdar = _class_source(mambaad, "PDARCSSD")
    forbidden = (
        "stage_delta",
        "deltas.append",
        "sources.append(stage_delta)",
        "stage_output - stage_input",
        "global_delta",
        "depth_global_delta",
    )
    for text in forbidden:
        if text in pdar:
            raise AssertionError(f"Forbidden PDAR delta logic remains: {text}")
    required = (
        "history = [x0]",
        "self.depth_mixers[stage_idx - 1](history)",
        "history.append(stage_output)",
        "self.final_depth_mixer(history)",
        "'depth_final_context': context_tokens",
        "add_outer_residual=False",
        "use_adaln=False",
    )
    for text in required:
        if text not in pdar:
            raise AssertionError(f"Missing PDAR full-history contract: {text}")
    print("PDAR full-history source contract: OK")

    lss = _class_source(mambaad, "LSSModule")
    required = (
        "local_kernel_sizes=(5, 7)",
        "local_dilations=(1, 1)",
        "self.local_effective_receptive_fields",
        "padding_5 = dilation_5 * (kernel_5 - 1) // 2",
        "padding_7 = dilation_7 * (kernel_7 - 1) // 2",
        "dilation=dilation_5",
        "dilation=dilation_7",
    )
    for text in required:
        if text not in lss:
            raise AssertionError(f"Missing configurable LSS local-view operation: {text}")

    progressive_required = (
        "local_receptive_field_schedule=None",
        "stage_kernel_sizes = [(3, 3) for _ in depths]",
        "(receptive_field - 1) // 2",
        "local_kernel_sizes=stage_kernel_sizes[idx]",
        "local_dilations=stage_dilations[idx]",
    )
    for text in progressive_required:
        if text not in pdar:
            raise AssertionError(f"Missing PDAR progressive-view wiring: {text}")
    print("PDAR progressive local-view contract: OK")

    mixer = _class_source(mambaad, "DepthAttentionResidual")
    required = (
        "nn.Linear(self.hidden_dim, 1, bias=False)",
        "V = torch.stack(list(sources), dim=0)",
        "K = self.norm(V)",
        "query = self.proj.weight.squeeze(0)",
        "'d, n b h w d -> n b h w'",
        "weights = logits.softmax(dim=0)",
        "'n b h w, n b h w d -> b h w d'",
    )
    for text in required:
        if text not in mixer:
            raise AssertionError(f"Missing official AttnRes operation: {text}")
    for text in ("query_proj", "value_proj"):
        if text in mixer:
            raise AssertionError(f"Forbidden AttnRes projection remains: {text}")
    print("official AttnRes mixer contract: OK")

    cssd = _class_source(mambaad, "CSSD")
    if "add_outer_residual=True" not in cssd:
        raise AssertionError("Baseline CSSD outer residual was changed.")
    if "use_adaln=True" not in cssd:
        raise AssertionError("Baseline CSSD AdaLN was not restored.")
    print("baseline CSSD residual and AdaLN contracts: OK")

    hss = _class_source(mambaad, "HSSBlock")
    required = (
        "use_adaln: bool = True",
        "if self.use_adaln:",
        "self.adaLN_modulation = nn.Sequential",
        "if self.use_adaln and c is not None:",
        "x_norm = x_norm * (1 + gamma_c) + beta_c",
    )
    for text in required:
        if text not in hss:
            raise AssertionError(f"Missing isolated AdaLN contract: {text}")
    print("isolated HSS AdaLN contract: OK")

    adapters = _read("model/modules/adapters.py")
    if "depth_global_delta" in adapters:
        raise AssertionError("Adapter still reads depth_global_delta.")
    if 'cssd_debug.get("depth_final_context", context_tokens)' not in adapters:
        raise AssertionError("Adapter does not read depth_final_context.")
    print("downstream final-context wiring: OK")

    clip = _read("model/clip_ad.py")
    config = _read("configs/clip_ad/clip_ad_supervised_mask_pdar_cssd.py")
    progressive_config = _read("configs/clip_ad/clip_ad_pdar_mvtec_supervised_to_visa.py")
    trainer = _read("trainer/clip_ad_trainer.py")
    if 'dbg_mamba_depth_w_f{idx}' not in clip:
        raise AssertionError("CLIP debug output does not use F-indexed weight names.")
    for idx in range(5):
        name = f"dbg_mamba_depth_w_f{idx}"
        if name not in config or name not in trainer:
            raise AssertionError(f"Missing final depth log: {name}")
    print("F0-F4 debug naming: OK")

    for text in (
        "local_receptive_field_schedule=(",
        "(3, 5)",
        "(5, 7)",
        "(7, 9)",
        "(9, 11)",
        "use_deformable_pool=False",
    ):
        if text not in progressive_config:
            raise AssertionError(f"Missing progressive-view config setting: {text}")
    print("MVTec-to-VisA progressive-view experiment config: OK")


if __name__ == "__main__":
    main()
