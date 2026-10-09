# -*- coding: utf-8 -*-
"""
تحلیل expert برای اسکریپت‌های new2 — از checkpoint (.pt) تا expert_analysis.json + plots.

مدل‌ها:
  moe_cif10, moe_cif100, moe_mnist, gdd_cif10, gdd_cif100, gdd_mnist,
  gdd_mnist_mag, moe_tiny_img200, gdd_tiny_img200

اجرا (تک مدل):
  python new2/expert_analysis.py --model moe_cif10 --ckpt new2/checkpoints/backbone_cifar10_hybrid_no_lbl_40.pt
  python new2/expert_analysis.py --model gdd_mnist --ckpt new2/mnist/C1_40/gdd_mnist_40.pt --out new2/analysis/gdd_mnist
  python new2/expert_analysis.py --model moe_mnist --ckpt new2/mnist/A1_40/moe_mnist_40.pt --out new2/analysis/moe_mnist
  python new2/expert_analysis.py --model gdd_mnist_mag --ckpt new2/mnist/D_42 --out new2/analysis/gdd_mnist_d42
  python new2/expert_analysis.py --model gdd_mnist_mag --ckpt new2/mnist/D_40 --out new2/analysis/gdd_mnist_d40

همهٔ مدل‌ها (با checkpoint پیش‌فرض):
  python new2/expert_analysis.py --all

فقط visualize روی JSON موجود:
  python new2/visualize_expert_analysis.py new2/analysis/moe_mnist/expert_analysis.json --kind moe
  python new2/visualize_expert_analysis.py new2/analysis/gdd_mnist/expert_analysis.json --kind gdd
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

NEW2 = Path(__file__).resolve().parent
ROOT = NEW2.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from expert_analysis_common import (  # noqa: E402
    effective_rank,
    expert_weight_vectors,
    image_low_level_features,
    linear_cka,
    pca_2d,
    routing_by_feature_bins,
)

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "moe_cif10": {
        "kind": "moe",
        "module": "moe_cif10",
        "model_cls": "CifarSparseMoEBackbone",
        "default_ckpt": "backbone_cifar10_hybrid_no_lbl_40.pt",
        "default_out": "analysis/moe_cif10",
    },
    "moe_cif100": {
        "kind": "moe",
        "module": "moe_cif100",
        "model_cls": "CifarSparseMoEBackbone",
        "default_ckpt": "moe_cifar100_lbl_0.00001_40.pt",
        "default_out": "analysis/moe_cif100",
    },
    "gdd_cif10": {
        "kind": "gdd",
        "module": "gdd_cif10",
        "model_cls": "CifarGDDSMoEBackbone",
        "default_ckpt": "gdds_moe_cifar10_S_layer_no_lbl_0.01_40.pt",
        "default_out": "analysis/gdd_cif10",
    },
    "gdd_cif100": {
        "kind": "gdd",
        "module": "gdd_cif100",
        "model_cls": "CifarGDDSMoEBackbone",
        "default_ckpt": "gdd_cif100_Sl_0.01__lb_0.0001_40.pt",
        "default_out": "analysis/gdd_cif100",
    },
    "moe_tiny_img200": {
        "kind": "moe",
        "module": "moe_tiny_img200",
        "model_cls": "CifarSparseMoEBackbone",
        "default_ckpt": "moe_tiny_img200_lbl_0.00001_40.pt",
        "default_out": "analysis/moe_tiny_img200",
    },
    "gdd_tiny_img200": {
        "kind": "gdd",
        "module": "gdd_tiny_img200",
        "model_cls": "CifarGDDSMoEBackbone",
        "default_ckpt": "gdd_tiny_img200_Sl_0.01__lb_0.0001_40.pt",
        "default_out": "analysis/gdd_tiny_img200",
    },
    "moe_mnist": {
        "kind": "moe",
        "module": "moe_mnist",
        "model_cls": "CifarSparseMoEBackbone",
        "default_ckpt": "moe_mnist_40.pt",
        "default_out": "analysis/moe_mnist",
        "ckpt_dirs": ["mnist/A1_40", "mnist/A1_41", "mnist/A1_42"],
    },
    "gdd_mnist": {
        "kind": "gdd",
        "module": "gdd_mnist_mag",
        "model_cls": "CifarGDDSMoEBackbone",
        "default_ckpt": "gdd_mnist_40.pt",
        "default_out": "analysis/gdd_mnist",
        "ckpt_dirs": ["mnist/C1_40", "mnist/C1_41", "mnist/C1_42"],
    },
    "gdd_mnist_mag": {
        "kind": "gdd",
        "module": "gdd_mnist_mag",
        "model_cls": "CifarGDDSMoEBackbone",
        "default_ckpt": "D_42.pt",
        "default_out": "analysis/gdd_mnist_d42",
        "ckpt_dirs": ["mnist/D_42", "mnist/D_40"],
    },
}


def _filter_dataclass(cls, d: Dict[str, Any]) -> Dict[str, Any]:
    valid = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in valid}


def _load_training_module(name: str):
    if str(NEW2) not in sys.path:
        sys.path.insert(0, str(NEW2))
    return importlib.import_module(name)


def cfg_from_ckpt(state: Dict[str, Any], mod) -> Any:
    BenchConfig = mod.BenchConfig
    MoEConfig = mod.MoEConfig
    raw = state.get("bench_config") or state.get("cfg")
    if raw is None:
        raise KeyError("checkpoint باید bench_config یا cfg داشته باشد")
    d = dict(raw)
    moe_raw = d.pop("moe", {})
    moe_d = _filter_dataclass(MoEConfig, moe_raw)
    bench_d = _filter_dataclass(BenchConfig, d)
    spatial_raw = bench_d.pop("spatial_moe", None)
    if spatial_raw is not None and isinstance(spatial_raw, dict):
        bench_d["spatial_moe"] = MoEConfig(**_filter_dataclass(MoEConfig, spatial_raw))
    return BenchConfig(moe=MoEConfig(**moe_d), **bench_d)


def state_dict_from_ckpt(state: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if state.get("best_model_state_dict"):
        return state["best_model_state_dict"]
    if state.get("state_dict"):
        return state["state_dict"]
    if state.get("model_state_dict"):
        return state["model_state_dict"]
    raise KeyError(
        "checkpoint باید یکی از best_model_state_dict / state_dict / model_state_dict را داشته باشد"
    )


def resolve_checkpoint_path(ckpt_path: str, mod) -> str:
    ap = os.path.abspath(os.path.expanduser(ckpt_path))
    if os.path.isfile(ap):
        return ap
    if os.path.isdir(ap):
        for name in ("training_last.pt", "best_model.pt"):
            p = os.path.join(ap, name)
            if os.path.isfile(p):
                return p
        best_glob = sorted(
            f for f in os.listdir(ap) if f.endswith("_best.pt") or f.endswith("best.pt")
        )
        if best_glob:
            return os.path.join(ap, best_glob[-1])
        pt_files = sorted(f for f in os.listdir(ap) if f.endswith(".pt"))
        if pt_files:
            return os.path.join(ap, pt_files[-1])
        raise FileNotFoundError(f"در پوشه {ap} فایل .pt پیدا نشد")
    norm = mod._normalize_save_path(ckpt_path, "training_last.pt")
    if norm and os.path.isfile(norm):
        return norm
    raise FileNotFoundError(f"checkpoint یافت نشد: {ckpt_path}")


def default_ckpt_path(model_name: str) -> Path:
    spec = MODEL_SPECS[model_name]
    fnames = spec.get("default_ckpts") or [spec["default_ckpt"]]
    extra = [NEW2 / p for p in spec.get("ckpt_dirs", [])]
    for base in (*extra, NEW2 / "checkpoints", NEW2, Path.cwd() / "checkpoints", Path.cwd()):
        for fname in fnames:
            p = base / fname
            if p.is_file():
                return p
        pts = sorted(base.glob("*.pt")) if base.is_dir() else []
        if extra and base in extra and pts:
            return pts[-1]
    return (extra[0] if extra else NEW2 / "checkpoints") / fnames[0]


def _log(msg: str) -> None:
    print(msg, flush=True)


def _forward_moe_layer(layer, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """SparseMoELayer در moe → 6 tuple؛ در gdd → 7 tuple (با expert_outputs)."""
    out = layer(h)
    if not isinstance(out, tuple) or len(out) < 4:
        raise TypeError(
            f"unexpected MoE layer return: {type(out)} "
            f"len={len(out) if isinstance(out, tuple) else '?'}"
        )
    return out[0], out[3]


@torch.no_grad()
def expert_outputs(layer, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = layer.norm(h)
    router_logits = layer.router(z)
    t = max(layer.cfg.router_temperature, 1e-6)
    scaled = router_logits / t
    topk_idx = scaled.topk(layer.cfg.top_k, dim=-1, stable=True).indices
    topk_val = scaled.gather(1, topk_idx)
    combine = F.softmax(topk_val, dim=-1)
    outs = torch.stack([expert(z) for expert in layer.experts], dim=0)
    return outs, topk_idx, combine


def _init_hidden(model, x: torch.Tensor):
    if model.cfg.use_hybrid_spatial_moe:
        tokens = model.encode_spatial(x)
        if model.spatial_pos_embed is not None:
            tokens = tokens + model.spatial_pos_embed
        b, t, d = tokens.shape
        h = tokens.reshape(b * t, d)
        n_sp = len(model.spatial_moe_layers)
        return h, b, t, n_sp
    h = model.encode(x)
    return h, x.shape[0], 1, 0


def _expand_image_features(feat: torch.Tensor, batch_size: int, tokens_per_image: int) -> np.ndarray:
    vals = feat.detach().cpu().numpy()
    if tokens_per_image <= 1:
        return vals
    return np.repeat(vals, tokens_per_image)


def _subsample_rows(x: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    n = x.shape[0]
    if n <= max_rows:
        return x
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:max_rows]
    return x.index_select(0, idx)


def _report_meta(model_name: str, spec: Dict[str, str], cfg, state: Dict[str, Any], ckpt_file: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "model_type": model_name,
        "model_kind": spec["kind"],
        "checkpoint": os.path.abspath(ckpt_file),
        "best_test_top1": state.get("best_eval_top1", state.get("best_test_top1")),
        "epoch": state.get("epochs_completed", state.get("epoch")),
        "seed": int(state.get("seed", cfg.seed)),
        "use_hybrid_spatial_moe": cfg.use_hybrid_spatial_moe,
        "use_load_balance_global": cfg.moe.use_load_balance,
        "load_balance_coef_global": cfg.moe.load_balance_coef,
        "use_load_balance_spatial": (
            cfg.spatial_moe.use_load_balance
            if cfg.spatial_moe is not None
            else cfg.moe.use_load_balance
        ),
        "moe_num_spatial_layers": cfg.moe_num_spatial_layers if cfg.use_hybrid_spatial_moe else 0,
        "moe_num_global_layers": (
            cfg.moe_num_global_layers if cfg.use_hybrid_spatial_moe else cfg.moe_num_layers
        ),
        "num_classes": cfg.num_classes,
    }
    if spec["kind"] == "gdd":
        meta.update(
            {
                "use_grad_div": cfg.use_grad_div,
                "grad_div_coef": cfg.grad_div_coef,
                "grad_div_global_only": getattr(cfg, "grad_div_global_only", None),
                "use_router_grad_div": getattr(cfg, "use_router_grad_div", False),
                "router_grad_div_coef": getattr(cfg, "router_grad_div_coef", None),
                "use_router_div": getattr(cfg, "use_router_div", False),
            }
        )
    return meta


def _render_plots(json_path: str, out_dir: str, kind: str) -> Optional[str]:
    if kind == "gdd":
        from visualize_expert_analysis_gdd import visualize_gdd

        return visualize_gdd(json_path, out_dir)
    from visualize_expert_analysis_backbone import visualize_backbone

    return visualize_backbone(json_path, out_dir)


@torch.no_grad()
def analyze_model(
    model_name: str,
    ckpt_path: str,
    out_dir: str,
    max_batches: int = 30,
    max_samples: int = 4096,
    device: str = "cuda",
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    skip_plots: bool = False,
) -> Dict[str, Any]:
    if model_name not in MODEL_SPECS:
        raise KeyError(f"مدل نامعتبر: {model_name}. گزینه‌ها: {', '.join(MODEL_SPECS)}")

    spec = MODEL_SPECS[model_name]
    mod = _load_training_module(spec["module"])
    ModelCls = getattr(mod, spec["model_cls"])

    t0 = time.time()
    ckpt_file = resolve_checkpoint_path(ckpt_path, mod)
    _log(f"[{model_name}] loading checkpoint: {ckpt_file}")
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    cfg = cfg_from_ckpt(state, mod)
    if data_dir is not None:
        cfg.data_dir = data_dir
    if batch_size is not None:
        cfg.batch_size = batch_size
    seed = int(state.get("seed", cfg.seed))
    mod.set_reproducibility(seed)
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    _log(f"      device={dev}, hybrid_spatial={cfg.use_hybrid_spatial_moe}, classes={cfg.num_classes}")

    model = ModelCls(cfg).to(dev)
    sd = state_dict_from_ckpt(state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        _log(f"      [warn] unexpected keys: {len(unexpected)}")
    if missing:
        _log(f"      [warn] missing keys: {len(missing)}")
    model.eval()

    _, test_ld = mod.loaders(cfg)
    n_layers = len(model.moe_layers)
    all_h: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    all_topk: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
    all_feat_edge: List[List[float]] = [[] for _ in range(n_layers)]
    all_feat_cvar: List[List[float]] = [[] for _ in range(n_layers)]
    num_images = 0

    _log(f"[{model_name}] collecting routing stats ({max_batches} batches)")
    for bi, (x, _y) in enumerate(test_ld):
        if bi >= max_batches:
            break
        x = x.to(dev)
        lf = image_low_level_features(x)
        num_images += x.shape[0]
        h, b, t, n_sp = _init_hidden(model, x)
        tokens_per_image = t
        for li, layer in enumerate(model.moe_layers):
            if cfg.use_hybrid_spatial_moe and li == n_sp:
                h = h.reshape(b, t, model.cfg.dim).mean(dim=1)
                tokens_per_image = 1
            h_in = h
            h, topk = _forward_moe_layer(layer, h)
            all_h[li].append(h_in.cpu())
            all_topk[li].append(topk.cpu())
            all_feat_edge[li].extend(
                _expand_image_features(lf["edge"], b, tokens_per_image).tolist()
            )
            all_feat_cvar[li].extend(
                _expand_image_features(lf["color_var"], b, tokens_per_image).tolist()
            )
        if (bi + 1) % max(1, max_batches // 5) == 0 or bi + 1 == max_batches:
            _log(f"      batch {bi + 1}/{max_batches} ({num_images} images)")

    report: Dict[str, Any] = {
        **_report_meta(model_name, spec, cfg, state, ckpt_file),
        "num_samples": num_images,
        "max_batches": max_batches,
        "max_samples_for_metrics": max_samples,
        "layers": {},
    }

    _log(f"[{model_name}] computing layer metrics")
    for li, layer in enumerate(model.moe_layers):
        layer_key = model.layer_prefix(li)
        num_experts = model.num_experts_for_layer(li)
        h_cat = torch.cat(all_h[li], dim=0)
        topk_cat = torch.cat(all_topk[li], dim=0).numpy()
        edge_np = np.array(all_feat_edge[li], dtype=np.float64)
        cvar_np = np.array(all_feat_cvar[li], dtype=np.float64)
        h_sub = _subsample_rows(h_cat, max_samples, seed + li).to(dev)
        outs, _, _ = expert_outputs(layer, h_sub)
        e = outs.shape[0]
        cos_mat = np.zeros((e, e), dtype=np.float64)
        cka_mat = np.zeros((e, e), dtype=np.float64)
        eff_ranks = []
        for i in range(e):
            yi = outs[i]
            eff_ranks.append(effective_rank(yi))
            for j in range(e):
                yj = outs[j]
                ci = F.normalize(yi, dim=-1)
                cj = F.normalize(yj, dim=-1)
                cos_mat[i, j] = float((ci * cj).sum(-1).mean().item())
                cka_mat[i, j] = linear_cka(yi, yj)
        wvec = expert_weight_vectors(layer)
        router_w = layer.router.weight.detach().cpu().numpy()
        layer_kind = (
            "spatial"
            if cfg.use_hybrid_spatial_moe and li < len(model.spatial_moe_layers)
            else "global"
        )
        report["layers"][layer_key] = {
            "layer_index": li,
            "layer_kind": layer_kind,
            "num_experts": num_experts,
            "top_k": layer.cfg.top_k,
            "expert_cosine_similarity": cos_mat.round(4).tolist(),
            "expert_linear_cka": cka_mat.round(4).tolist(),
            "expert_effective_rank": [round(r, 2) for r in eff_ranks],
            "weight_pca_2d": pca_2d(wvec).tolist(),
            "router_weight_heatmap": router_w.round(4).tolist(),
            "router_weight_pca_2d": pca_2d(router_w).tolist(),
            "routing_by_edge_quartile": routing_by_feature_bins(edge_np, topk_cat, num_experts),
            "routing_by_color_var_quartile": routing_by_feature_bins(cvar_np, topk_cat, num_experts),
            "mean_expert_cosine_offdiag": float(cos_mat[~np.eye(e, dtype=bool)].mean()),
        }

    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "expert_analysis.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _log(f"[{model_name}] wrote {json_path}")

    html_path = None
    if not skip_plots:
        plots_dir = os.path.join(out_dir, "plots")
        _log(f"[{model_name}] generating plots")
        html_path = _render_plots(json_path, plots_dir, spec["kind"])

    _log(f"[{model_name}] done in {time.time() - t0:.1f}s")
    report["_json_path"] = json_path
    report["_html_report"] = html_path
    return report


def run_all(
    max_batches: int,
    max_samples: int,
    device: str,
    data_dir: Optional[str],
    skip_plots: bool,
) -> List[Dict[str, Any]]:
    results = []
    for name in MODEL_SPECS:
        ckpt = default_ckpt_path(name)
        out = str(NEW2 / MODEL_SPECS[name]["default_out"])
        if not ckpt.is_file():
            _log(f"[skip] {name}: checkpoint not found — {ckpt}")
            continue
        try:
            rep = analyze_model(
                name,
                str(ckpt),
                out,
                max_batches=max_batches,
                max_samples=max_samples,
                device=device,
                data_dir=data_dir,
                skip_plots=skip_plots,
            )
            results.append({"model": name, "ok": True, "out": out, "html": rep.get("_html_report")})
        except Exception as exc:
            _log(f"[error] {name}: {exc}")
            results.append({"model": name, "ok": False, "error": str(exc)})
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Expert analysis for new2 models")
    ap.add_argument(
        "--model",
        choices=sorted(MODEL_SPECS),
        help="model script name (required unless --all)",
    )
    ap.add_argument("--all", action="store_true", help="run all registered models with default checkpoints")
    ap.add_argument("--ckpt", default=None, help="path to .pt checkpoint (folder also ok)")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--max-batches", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=4096)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data-dir", default=None, help="override cfg.data_dir (Tiny-ImageNet)")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--skip-plots", action="store_true")
    args = ap.parse_args()

    if args.all:
        summary = run_all(
            args.max_batches,
            args.max_samples,
            args.device,
            args.data_dir,
            args.skip_plots,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if not args.model:
        ap.error("یکی از --model یا --all لازم است")

    ckpt = args.ckpt or str(default_ckpt_path(args.model))
    out = args.out or str(NEW2 / MODEL_SPECS[args.model]["default_out"])
    rep = analyze_model(
        args.model,
        ckpt,
        out,
        max_batches=args.max_batches,
        max_samples=args.max_samples,
        device=args.device,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        skip_plots=args.skip_plots,
    )
    print(
        json.dumps(
            {
                "model": args.model,
                "json": rep["_json_path"],
                "html_report": rep.get("_html_report"),
                "layers": list(rep["layers"].keys()),
                "num_samples": rep["num_samples"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
