# Copyright 2025
# -*- coding: utf-8 -*-
"""
GDDS-MoE — Expert Diversity via Gradient Direction Separation (Tiny ImageNet-200 / Colab).

ایده: تخصص رفتاری با جدا کردن **جهت گرادیان یادگیری** expertها (GDDS) و
اختیاری **learning response** router (GRD):

  L_total = L_task + λ_lb·L_load_balance + λ_g·L_GDDS [+ λ_r·L_GRD]

  GDDS: L_div = mean_{i≠j} cos²(∇_{W_i} L_i, ∇_{W_j} L_j)  (global MoE)

معماری: Hybrid Spatial MoE → GAP → Global MoE (مثل gdd_cif10_v1).

اجرا:
  python gdd_tiny_img200.py
  # Colab: %run gdd_tiny_img200.py
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
  """CKA خطی بین دو ماتریس [N, D]."""
  x = x - x.mean(0, keepdim=True)
  y = y - y.mean(0, keepdim=True)
  hsic = (y.t() @ x).pow(2).sum()
  norm = (x.t() @ x).norm() * (y.t() @ y).norm()
  if float(norm.item()) < 1e-12:
    return 0.0
  return float((hsic / norm).item())


def _expert_weight_vectors(layer: "SparseMoELayer") -> np.ndarray:
  vecs = []
  for expert in layer.experts:
    parts = [p.detach().cpu().flatten() for p in expert.parameters()]
    vecs.append(torch.cat(parts).numpy())
  return np.stack(vecs, axis=0)


def _pca_2d(matrix: np.ndarray) -> np.ndarray:
  x = matrix - matrix.mean(0, keepdims=True)
  _, _, vt = np.linalg.svd(x, full_matrices=False)
  return x @ vt[:2].T


# =============================================================================
# 1) تنظیمات MoE (همهٔ «بهبودها» اینجا خاموش/روشن می‌شوند)
# =============================================================================


@dataclass
class MoEConfig:
  """همهٔ رفتار MoE یک لایه؛ برای baseline همهٔ aux را False و coef=0 بگذار."""

  num_experts: int = 8
  top_k: int = 2
  expert_hidden: int = 512
  expert_dropout: float = 0.1

  router_temperature: float = 1.0
  router_noise_std: float = 0.0
  # اگر True: فقط برای انتخاب top-k از logits جدا استفاده می‌شود (مسیر کمکی)
  router_detach_for_topk: bool = False

  # ترکیب وزن‌ها روی اسلات‌های top-k
  combine_method: str = "softmax"  # فعلاً فقط "softmax"

  # ظرفیت هر اکسپرت در هر بچ: ceil(factor * B * top_k / E)؛ None = بدون سقف
  capacity_factor: Optional[float] = None

  # residual: y = x + scale * moe_out
  moe_residual_scale: float = 1.0

  # --- لاس‌های کمکی (همه اختیاری؛ برای pure baseline همه False / 0) ---
  use_load_balance: bool = False
  load_balance_coef: float = 0.01

  use_z_loss: bool = False
  z_loss_coef: float = 1e-3

  # آنتروپی روی softmax کامل روتِر (نه فقط top-k) — ممکن است با دقت trade-off داشته باشد
  use_router_entropy_aux: bool = False
  router_entropy_coef: float = 0.0


@dataclass
class BenchConfig:
  """آموزش + داده + لاگ."""

  moe: MoEConfig = field(default_factory=MoEConfig)
  num_classes: int = 200
  image_size: int = 64
  dim: int = 384
  moe_num_layers: int = 2
  batch_size: int = 128
  epochs: int = 30
  # پیش‌فرض‌های پایدارتر برای MoE (مقایسه با LM معمولی)
  lr: float = 2e-3
  weight_decay: float = 0.03
  adam_betas: Tuple[float, float] = (0.9, 0.98)
  adam_eps: float = 1e-8
  seed: int = 41
  num_workers: int = 0
  device: str = "cuda"
  data_dir: str = "./data"
  local_tiny_imagenet: Optional[str] = "tiny-imagenet-200.zip"
  train_random_crop_pad: int = 8
  train_hflip: bool = True
  train_randaugment_ops: int = 2
  train_randaugment_magnitude: int = 4
  train_label_smoothing: float = 0.1
  log_json: Optional[str] = "sparse_moe_bench_colab.json"
  # فایل متنی جداگانه برای ذخیرهٔ خروجی کنسول؛ None = خودکار از log_json.
  log_console: Optional[str] = None
  # تعداد رقم اعشار برای floatها در JSON لاگ و چاپ config (جلوگیری از خروجی‌های مثل 0.4355200000095367)
  log_json_float_decimals: int = 4
  # اگر مسیر فایل (یا پوشه) باشد: **پایان آموزش** بهترین وزنها ذخیره می‌شود (فقط state_dict + متادیتا).
  checkpoint_path: Optional[str] = None
  # پوشهٔ ذخیرهٔ بهترین مدل؛ پس از هر بهبود test_top1 بازنویسی می‌شود. None = غیرفعال.
  checkpoint_dir: Optional[str] = "checkpoints"
  # نام فایل checkpoint؛ None = خودکار از log_json و seed.
  checkpoint_filename: Optional[str] = None
  # فایل چک‌پوینت کامل آموزش: بعد از **هر اپوک** بازنویسی می‌شود (مدل، optimizer، scheduler، اپوک، RNG، …).
  # None = غیرفعال (فقط checkpoint سبک در checkpoint_dir ذخیره می‌شود).
  training_checkpoint_path: Optional[str] = None
  # اگر این مسیر فایل وجود داشته باشد، آموزش از همان ادامه پیدا می‌کند (معمولاً همان `training_checkpoint_path`).
  resume_training_path: Optional[str] = None
  # اگر عدد باشد: بعد از این تعداد اپوک متوالی بدون بهبود متوقف می‌شود.
  early_stop_patience: Optional[int] = 20
  early_stop_min_delta: float = 1e-4
  lr_warmup_epochs: int = 0
  lr_min: float = 1e-6
  # متریک‌های specialization (۵ متریک اصلی)
  spec_metric_enabled: bool = True
  # None = کل test set برای ablation؛ عدد = حداکثر batch (سریع‌تر)
  spec_ablation_max_batches: Optional[int] = 50

  # --- GDDS: Gradient Direction Diversity Separation ---
  use_grad_div: bool = True
  grad_div_coef: float = 0.003
  # در hybrid: هر flag یک خانواده لایه را فعال می‌کند؛ اگر هر دو True باشند هر دو اعمال می‌شوند.
  grad_div_global_only: bool = True
  grad_div_spatial_only: bool = False
  # حداکثر توکن برای محاسبهٔ L_div (کاهش هزینهٔ حافظه)
  grad_div_max_tokens: int = 128

  # --- Router activation diversity (legacy; برای ablation) ---
  use_router_div: bool = False
  router_div_coef: float = 0.003
  router_div_max_tokens: int = 256

  # --- GRD: Gradient Response Diversity (router ← learning dynamics) ---
  use_router_grad_div: bool = True
  router_grad_div_coef: float = 0.001
  router_grad_div_max_tokens: int = 256
  router_grad_div_global_only: bool = True
  router_grad_div_detach_upstream: bool = True
  router_grad_div_detach_sim: bool = True

  # --- Hybrid: Spatial MoE (قبل از GAP) → GAP → Global MoE ---
  use_hybrid_spatial_moe: bool = False
  spatial_moe: Optional[MoEConfig] = None
  moe_num_spatial_layers: int = 1
  moe_num_global_layers: int = 2
  use_spatial_pos_embed: bool = True
  # عرض خروجی stage3 بک‌بون (قبل از embed → MoE)
  feature_channels: int = 384
  # تعداد BasicBlock در stage3 (پیش‌فرض قدیمی = 2)
  backbone_stage3_depth: int = 3


def _clone_moe_config(cfg: MoEConfig, **overrides: Any) -> MoEConfig:
  d = asdict(cfg)
  d.update(overrides)
  valid = {f.name for f in fields(MoEConfig)}
  return MoEConfig(**{k: v for k, v in d.items() if k in valid})


def resolve_hybrid_moe_configs(cfg: BenchConfig) -> Tuple[MoEConfig, MoEConfig]:
  """تنظیمات spatial/global؛ spatial_moe=None → پیش‌فرض مناسب spatial."""
  global_moe = cfg.moe
  if cfg.spatial_moe is not None:
    return cfg.spatial_moe, global_moe
  cap = global_moe.capacity_factor
  spatial_cap = 1.5 if cap is not None else None
  spatial_moe = _clone_moe_config(
      global_moe,
      capacity_factor=spatial_cap,
      router_noise_std=max(0.02, float(global_moe.router_noise_std)),
  )
  return spatial_moe, global_moe


def as_bench_config(cfg: Union[BenchConfig, MoEConfig]) -> BenchConfig:
  """اگر فقط MoEConfig دارید، برای run_benchmark به BenchConfig تبدیل می‌شود."""
  if isinstance(cfg, BenchConfig):
    return cfg
  if isinstance(cfg, MoEConfig):
    return BenchConfig(moe=cfg)
  raise TypeError(f"انتظار BenchConfig یا MoEConfig؛ دریافت {type(cfg)!r}")


def print_full_config(cfg: BenchConfig) -> None:
  d = asdict(cfg)
  nd = max(int(cfg.log_json_float_decimals), 0)
  print("=" * 72)
  print(json.dumps(round_json_floats(d, nd), indent=2, ensure_ascii=False, default=str))
  print("=" * 72, flush=True)


def round_json_floats(obj: Any, ndigits: int = 4) -> Any:
  """گرد کردن floatها برای JSON لاگ و چاپ config."""
  if isinstance(obj, (float, np.floating)) and not isinstance(obj, bool):
    x = float(obj)
    return round(x, ndigits) if math.isfinite(x) else x
  if isinstance(obj, dict):
    return {k: round_json_floats(v, ndigits) for k, v in obj.items()}
  if isinstance(obj, list):
    return [round_json_floats(v, ndigits) for v in obj]
  if isinstance(obj, tuple):
    return tuple(round_json_floats(v, ndigits) for v in obj)
  return obj


_TRAINING_CKPT_VERSION = 1


def _normalize_save_path(path: Optional[str], default_filename: str) -> Optional[str]:
  if not path:
    return None
  ap = os.path.abspath(os.path.expanduser(path))
  if os.path.isdir(ap):
    return os.path.join(ap, default_filename)
  return ap


def _cuda_rng_snapshot() -> List[torch.Tensor]:
  out: List[torch.Tensor] = []
  if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
      out.append(torch.cuda.get_rng_state(i))
  return out


def _restore_cuda_rng(states: List[torch.Tensor]) -> None:
  if not torch.cuda.is_available() or not states:
    return
  n = min(len(states), torch.cuda.device_count())
  for i in range(n):
    s = states[i]
    torch.cuda.set_rng_state(s.cpu() if s.is_cuda else s, i)


def _optimizer_state_to_device(opt: torch.optim.Optimizer, device: torch.device) -> None:
  for st in opt.state.values():
    for k in list(st.keys()):
      if torch.is_tensor(st[k]):
        st[k] = st[k].to(device)


def save_training_checkpoint(
    path: str,
    cfg: BenchConfig,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    epochs_completed: int,
    best_eval_top1: float,
    best_model_state: Optional[Dict[str, torch.Tensor]],
    epochs_no_improve: int,
) -> None:
  parent = os.path.dirname(os.path.abspath(path))
  if parent:
    os.makedirs(parent, exist_ok=True)
  payload: Dict[str, Any] = {
      "version": _TRAINING_CKPT_VERSION,
      "epochs_completed": int(epochs_completed),
      "bench_config": asdict(cfg),
      "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
      "optimizer_state_dict": opt.state_dict(),
      "scheduler_state_dict": sched.state_dict(),
      "best_eval_top1": float(best_eval_top1),
      "epochs_no_improve": int(epochs_no_improve),
      "torch_rng_state": torch.get_rng_state(),
      "numpy_state": np.random.get_state(),
      "python_rng_state": random.getstate(),
      "cuda_rng_states": _cuda_rng_snapshot(),
  }
  if best_model_state is not None:
    payload["best_model_state_dict"] = {
        k: v.detach().cpu().clone() for k, v in best_model_state.items()
    }
  torch.save(payload, path)


def load_training_checkpoint(
    path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    dev: torch.device,
    cfg: BenchConfig,
) -> Tuple[int, float, Optional[Dict[str, torch.Tensor]], int]:
  ck = torch.load(path, map_location="cpu", weights_only=False)
  lc = ck.get("bench_config")
  if isinstance(lc, dict):
    if (
        lc.get("dim") != cfg.dim
        or lc.get("num_classes") != cfg.num_classes
        or lc.get("feature_channels") != cfg.feature_channels
        or lc.get("backbone_stage3_depth") != cfg.backbone_stage3_depth
    ):
      print(
          "هشدار: dim / num_classes / feature_channels / backbone_stage3_depth "
          "با چک‌پوینت فرق دارد؛ بارگذاری ممکن است نامعتبر باشد.",
          flush=True,
      )
    if lc.get("epochs") != cfg.epochs:
      print(
          f"هشدار: در چک‌پوینت epochs={lc.get('epochs')} ولی cfg فعلی epochs={cfg.epochs}؛ "
          "برای scheduler بهتر است همان تعداد اپوک اولیه را نگه دارید یا از صفر با وزن‌های ذخیره‌شده شروع کنید.",
          flush=True,
      )
  model.load_state_dict(ck["model_state_dict"])
  model.to(dev)
  opt.load_state_dict(ck["optimizer_state_dict"])
  _optimizer_state_to_device(opt, dev)
  sched.load_state_dict(ck["scheduler_state_dict"])
  epochs_completed = int(ck["epochs_completed"])
  best_eval_top1 = float(ck.get("best_eval_top1", float("-inf")))
  bms = ck.get("best_model_state_dict")
  best_model_state: Optional[Dict[str, torch.Tensor]]
  if bms is not None:
    best_model_state = {k: v.clone() for k, v in bms.items()}
  else:
    best_model_state = None
  epochs_no_improve = int(ck.get("epochs_no_improve", 0))
  if "torch_rng_state" in ck:
    torch.set_rng_state(ck["torch_rng_state"])
  if "numpy_state" in ck:
    np.random.set_state(ck["numpy_state"])
  if "python_rng_state" in ck:
    random.setstate(ck["python_rng_state"])
  _restore_cuda_rng(ck.get("cuda_rng_states") or [])
  return epochs_completed, best_eval_top1, best_model_state, epochs_no_improve


def set_reproducibility(seed: int) -> torch.Generator:
    """Seed یکسان برای random / numpy / torch؛ cudnn قطعی؛ تا حد ممکن خروجی تکرارپذیر."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:
        os.environ["FORCE_DETERMINISTIC_CALLBACKS"] = "1"
        print(
            f"هشدار: torch.use_deterministic_algorithms فعال نشد ({exc}). "
            "برای تکرارپذیری کامل CUDA، FORCE_DETERMINISTIC_CALLBACKS=1 تنظیم شد.",
            flush=True,
        )

    g = torch.Generator()
    g.manual_seed(seed)

    return g


def _dataloader_seed_worker(worker_id: int) -> None:
  w_seed = (torch.initial_seed() % 2**32) + worker_id
  np.random.seed(w_seed)
  random.seed(w_seed)


def adamw_param_groups(model: nn.Module, weight_decay: float) -> List[Dict[str, Any]]:
  """بدون weight decay روی بایاس، نرمال‌سازی‌ها و پارامترهای ۱بعدی."""
  decay: List[nn.Parameter] = []
  no_decay: List[nn.Parameter] = []
  for name, p in model.named_parameters():
    if not p.requires_grad:
      continue
    if p.ndim == 1 or "bias" in name or "norm" in name:
      no_decay.append(p)
    else:
      decay.append(p)
  return [
      {"params": decay, "weight_decay": weight_decay},
      {"params": no_decay, "weight_decay": 0.0},
  ]


# =============================================================================
# 1.5) ذخیرهٔ خروجی کنسول در فایل جداگانه (Tee)
# =============================================================================


class _ConsoleTee:
  """هرچه به stream اصلی نوشته شود، هم‌زمان در فایل هم نوشته می‌شود."""

  def __init__(self, stream: Any, fh: Any) -> None:
    self._stream = stream
    self._fh = fh

  def write(self, data: str) -> int:
    self._stream.write(data)
    try:
      self._fh.write(data)
    except (ValueError, OSError):
      pass
    return len(data)

  def flush(self) -> None:
    self._stream.flush()
    try:
      self._fh.flush()
    except (ValueError, OSError):
      pass

  def __getattr__(self, name: str) -> Any:
    return getattr(self._stream, name)


def _default_console_path(cfg: BenchConfig) -> Optional[str]:
  if cfg.log_console:
    return cfg.log_console
  if cfg.log_json:
    base, _ = os.path.splitext(cfg.log_json)
    return base + "_console.log"
  return None


@contextlib.contextmanager
def console_logging(path: Optional[str]):
  if not path:
    yield None
    return
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  fh = open(path, "a", encoding="utf-8")
  old_out, old_err = sys.stdout, sys.stderr
  sys.stdout = _ConsoleTee(old_out, fh)
  sys.stderr = _ConsoleTee(old_err, fh)
  try:
    fh.write(
        "\n" + "#" * 72 + "\n"
        f"# run start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        + "#" * 72 + "\n"
    )
    fh.flush()
    yield path
  finally:
    sys.stdout = old_out
    sys.stderr = old_err
    try:
      fh.write(
          "#" * 72 + "\n"
          f"# run end: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
          + "#" * 72 + "\n"
      )
      fh.flush()
      fh.close()
    except (ValueError, OSError):
      pass


# =============================================================================
# 2) Expert
# =============================================================================


class ExpertMLP(nn.Module):
  def __init__(self, dim: int, hidden: int, dropout: float) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, dim),
    )
    linear_layers = [m for m in self.net if isinstance(m, nn.Linear)]
    for i, m in enumerate(linear_layers):
      if i == len(linear_layers) - 1:
        nn.init.zeros_(m.weight)
      else:
        nn.init.xavier_uniform_(m.weight, gain=1.0)
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)

# =============================================================================
# 3) لاس‌های کمکی ماژولار
# =============================================================================


def load_balance_loss(
    router_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    num_experts: int,
    top_k: int,
) -> torch.Tensor:
  """Switch/GShard-style: همان logitsی که برای routing استفاده شده (مثلاً scaled+noise)."""
  batch = float(router_logits.shape[0])
  p = F.softmax(router_logits, dim=-1)
  importance = p.mean(dim=0)
  importance = importance / (importance.sum() + 1e-20)
  load = torch.zeros(num_experts, device=router_logits.device, dtype=p.dtype)
  for i in range(top_k):
    load = load + F.one_hot(topk_indices[:, i], num_experts).float().sum(dim=0)
  load = load / batch
  return (importance * load).sum() * num_experts


@dataclass
class MoEAuxMetrics:
  """Routing metrics for benchmark logging."""

  load_balance_loss_sum: float = 0.0
  n_batches: int = 0
  n_tokens: int = 0
  slot_k: int = 0
  importance_sum: Optional[torch.Tensor] = None
  expert_hits: Optional[torch.Tensor] = None
  routing_entropy_sum: float = 0.0
  num_experts_tracked: int = 0

  def to_dict(self) -> Dict[str, float]:
    d: Dict[str, float] = {}
    if self.n_batches == 0 or self.n_tokens == 0:
      return d
    lb = self.load_balance_loss_sum / self.n_batches
    H = self.routing_entropy_sum / self.n_batches
    d["load_balance_loss"] = lb
    E = self.num_experts_tracked
    if E <= 0 and self.importance_sum is not None:
      E = int(self.importance_sum.shape[0])
    if E > 0:
      d["load_balance_excess"] = float(lb - 1.0)
      d["load_balance_excess_norm"] = float((lb - 1.0) / max(float(E - 1), 1.0))
      d["routing_mean_entropy"] = float(H)
      d["routing_entropy_norm"] = float(H / max(math.log(float(E)), 1e-8))
    if self.importance_sum is not None and self.expert_hits is not None:
      e = int(self.importance_sum.shape[0])
      p_mean = (self.importance_sum / float(self.n_tokens)).cpu()
      denom = float(self.n_tokens) * max(float(self.slot_k), 1.0)
      hits = (self.expert_hits / denom).cpu()
      p_uniform = 1.0 / e
      d["expert_load_imbalance_l2"] = float(((hits - hits.mean()) ** 2).sum().sqrt().item())
      d["expert_importance_imbalance_l2"] = float(
          ((p_mean - p_mean.mean()) ** 2).sum().sqrt().item()
      )
      d["deviation_from_uniform_l2"] = float(((p_mean - p_uniform) ** 2).sum().sqrt().item())
    return d

  def _ensure_buffers(self, num_experts: int, device: torch.device) -> None:
    if self.importance_sum is None:
      self.importance_sum = torch.zeros(num_experts, device=device, dtype=torch.float32)
    if self.expert_hits is None:
      self.expert_hits = torch.zeros(num_experts, device=device, dtype=torch.float32)

  @torch.no_grad()
  def accumulate(
      self,
      router_logits: torch.Tensor,
      topk_indices: torch.Tensor,
      lb_loss: torch.Tensor,
  ) -> None:
    b, e = router_logits.shape
    self.num_experts_tracked = int(e)
    p = F.softmax(router_logits, dim=-1)
    self._ensure_buffers(e, router_logits.device)
    assert self.importance_sum is not None and self.expert_hits is not None
    self.importance_sum = self.importance_sum + p.sum(0)
    k = int(topk_indices.shape[1])
    if self.slot_k == 0:
      self.slot_k = k
    slot_mass = torch.zeros(e, device=router_logits.device, dtype=p.dtype)
    for ki in range(k):
      slot_mass = slot_mass + F.one_hot(topk_indices[:, ki], e).float().sum(0)
    self.expert_hits = self.expert_hits + slot_mass
    ent = float((-(p * p.clamp_min(1e-8).log()).sum(-1).mean()).item())
    self.routing_entropy_sum += ent
    self.load_balance_loss_sum += float(lb_loss.item())
    self.n_batches += 1
    self.n_tokens += int(b)


def _mean_moe_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
  if not dicts:
    return {}
  keys = dicts[0].keys()
  n = float(len(dicts))
  return {k: float(sum(d.get(k, 0.0) for d in dicts) / n) for k in keys}






def _train_aux_for_log(aux: Dict[str, Any]) -> Dict[str, Any]:
  """Aux stats for JSON/console without grad-div diagnostic keys."""
  omit_exact = frozenset({"grad_div_total", "router_grd_total"})
  omit_suffix = (
      "_grad_div",
      "_grad_cos_abs",
      "_router_grd",
      "_router_grd_resp_cos",
      "_router_grd_pair_prob",
  )

  def _skip(k: str) -> bool:
    if k in omit_exact:
      return True
    return any(k.endswith(s) for s in omit_suffix)

  return {k: v for k, v in aux.items() if not _skip(k)}

def _normalize_log_json_path(path: Optional[str]) -> Optional[str]:
  """مسیر لاگ جزئیات: یک فایل `.json` (نه jsonl)."""
  if not path:
    return None
  base, ext = os.path.splitext(path)
  if ext.lower() == ".jsonl":
    return base + ".json"
  if not ext:
    return path + ".json"
  return path


def _accumulate_layer_dispatch(
    agg: Dict[str, Any],
    st: Dict[str, Any],
    num_experts: int,
    layer_idx: int = 0,
) -> None:
  nf = int(st.get("n_flat", 0))
  if nf <= 0:
    return
  agg["n_flat"] = int(agg.get("n_flat", 0)) + nf
  dropped = int(st.get("dropped_assignments", 0))
  agg["dropped_assignments"] = int(agg.get("dropped_assignments", 0)) + dropped
  agg["expert_calls"] = int(agg.get("expert_calls", 0)) + int(st.get("expert_calls", 0))
  agg["active_experts_fwd_sum"] = float(agg.get("active_experts_fwd_sum", 0.0)) + float(
      st.get("active_experts_ratio_fwd", 0.0)
  )
  agg["n_stat_forwards"] = int(agg.get("n_stat_forwards", 0)) + 1

  ln_flat: Dict[int, int] = agg.setdefault("layer_n_flat", {})
  ln_flat[layer_idx] = int(ln_flat.get(layer_idx, 0)) + nf
  ln_drop: Dict[int, int] = agg.setdefault("layer_dropped", {})
  ln_drop[layer_idx] = int(ln_drop.get(layer_idx, 0)) + dropped
  ln_calls: Dict[int, int] = agg.setdefault("layer_expert_calls", {})
  ln_calls[layer_idx] = int(ln_calls.get(layer_idx, 0)) + int(st.get("expert_calls", 0))

  kept_list = st.get("tokens_per_expert_kept")
  if kept_list is None:
    kept_list = [0] * num_experts
  tvec = torch.tensor(kept_list, dtype=torch.float64)
  layer_sums: Dict[int, torch.Tensor] = agg.setdefault("layer_tokens_sum", {})
  if layer_idx not in layer_sums:
    layer_sums[layer_idx] = torch.zeros(num_experts, dtype=torch.float64)
  buf = layer_sums[layer_idx]
  if buf.numel() < num_experts:
    layer_sums[layer_idx] = torch.zeros(num_experts, dtype=torch.float64)
    layer_sums[layer_idx][: buf.numel()] = buf
  n = min(int(tvec.numel()), num_experts)
  layer_sums[layer_idx][:n] = layer_sums[layer_idx][:n] + tvec[:n]


def _layer_dispatch_block(
    tk: torch.Tensor,
    num_experts: int,
    n_flat: int,
    dropped: int,
    expert_calls: int,
    ar_fwd_mean: float,
) -> Dict[str, Any]:
  if tk.numel() < num_experts:
    tk = torch.cat([tk, torch.zeros(num_experts - tk.numel(), dtype=tk.dtype)])
  elif tk.numel() > num_experts:
    tk = tk[:num_experts]
  denom = float(n_flat) if n_flat > 0 else 1.0
  n_kept = max(n_flat - dropped, 0)
  return {
      "dispatch_n_flat": float(n_flat),
      "dropped_assignments": float(dropped),
      "expert_calls": float(expert_calls),
      "drop_rate": float(dropped) / denom,
      "capacity_usage_ratio": float(n_kept) / denom,
      "tokens_per_expert": [float(x) for x in tk.tolist()],
      "mean_tokens_per_expert": float(tk.mean().item()),
      "std_tokens_per_expert": float(tk.std(unbiased=False).item()),
      "active_experts_ratio": (
          float((tk > 0).sum().item()) / float(num_experts) if num_experts > 0 else 0.0
      ),
      "active_experts_ratio_fwd_mean": float(ar_fwd_mean),
  }


def _finalize_dispatch_metrics(
    agg: Dict[str, Any],
    layer_specs: List[Tuple[int, str, int]],
) -> Dict[str, Any]:
  """متریک dispatch سراسری + تفکیک هر لایه MoE (S0, G0, …)."""
  n_flat = max(int(agg.get("n_flat", 0)), 0)
  dropped = max(int(agg.get("dropped_assignments", 0)), 0)
  ec = max(int(agg.get("expert_calls", 0)), 0)
  n_kept = max(n_flat - dropped, 0)
  denom = float(n_flat) if n_flat > 0 else 1.0
  n_stat = max(int(agg.get("n_stat_forwards", 0)), 0)
  ar_fwd_mean = (
      float(agg.get("active_experts_fwd_sum", 0.0)) / float(n_stat) if n_stat > 0 else float("nan")
  )

  layer_sums: Dict[int, torch.Tensor] = agg.get("layer_tokens_sum") or {}
  ln_flat: Dict[int, int] = agg.get("layer_n_flat") or {}
  ln_drop: Dict[int, int] = agg.get("layer_dropped") or {}
  ln_calls: Dict[int, int] = agg.get("layer_expert_calls") or {}

  layers_out: Dict[str, Any] = {}
  for li, prefix, ne in layer_specs:
    tk = layer_sums.get(li, torch.zeros(ne, dtype=torch.float64)).float()
    layers_out[prefix] = _layer_dispatch_block(
        tk,
        ne,
        int(ln_flat.get(li, 0)),
        int(ln_drop.get(li, 0)),
        int(ln_calls.get(li, 0)),
        ar_fwd_mean,
    )

  if layer_specs:
    top_tokens = layers_out[layer_specs[0][1]]
  else:
    top_tokens = _layer_dispatch_block(
        torch.zeros(1, dtype=torch.float32),
        1,
        n_flat,
        dropped,
        ec,
        ar_fwd_mean,
    )

  return {
      "dispatch_n_flat": float(n_flat),
      "dropped_assignments": float(dropped),
      "expert_calls": float(ec),
      "drop_rate": float(dropped) / denom,
      "capacity_usage_ratio": float(n_kept) / denom,
      "tokens_per_expert": top_tokens["tokens_per_expert"],
      "mean_tokens_per_expert": top_tokens["mean_tokens_per_expert"],
      "std_tokens_per_expert": top_tokens["std_tokens_per_expert"],
      "active_experts_ratio": top_tokens["active_experts_ratio"],
      "active_experts_ratio_fwd_mean": float(ar_fwd_mean),
      "layers": layers_out,
  }


def _write_best_epoch_log(
    log_path: str,
    epoch_row: Dict[str, Any],
    bench_config: Dict[str, Any],
    float_decimals: int = 4,
) -> None:
  """یک فایل JSON با جزئیات اپوک best (بازنویسی کامل؛ همان لحظهٔ بهبود)."""
  os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
  nd = max(int(float_decimals), 0)
  payload = {
      "log_format": "best_epoch_v1",
      "bench_config": bench_config,
      **epoch_row,
  }
  with open(log_path, "w", encoding="utf-8") as f:
    json.dump(
        round_json_floats(payload, nd),
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    f.write("\n")

def z_loss_router(router_logits):
    z = torch.logsumexp(router_logits, dim=-1)
    return (z ** 2).mean()


def router_entropy_aux(router_logits: torch.Tensor) -> torch.Tensor:
  p = F.softmax(router_logits, dim=-1)
  return (-(p * p.clamp_min(1e-8).log()).sum(-1).mean())


def build_aux_losses(
    router_logits: torch.Tensor,
    topk_indices: torch.Tensor,
    cfg: MoEConfig,
    routing_logits: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
  """فقط تنسور خام؛ ضرب در coef در `sum_weighted_aux_fixed`.
  routing_logits: اگر ست باشد، برای load balance همان تنسوری است که top-k روی آن زده شده
  (مثلاً بعد از temp/noise) تا لاس و لاگ با مسیر واقعی هم‌خوان باشند.
  """
  out: Dict[str, torch.Tensor] = {}
  lb_logits = routing_logits if routing_logits is not None else router_logits
  if cfg.use_load_balance:
    out["load_balance"] = load_balance_loss(
        lb_logits, topk_indices, cfg.num_experts, cfg.top_k
    )
  if cfg.use_z_loss:
    out["z"] = z_loss_router(router_logits)
  if cfg.use_router_entropy_aux:
    out["router_entropy"] = router_entropy_aux(router_logits)
  return out


def sum_weighted_aux_fixed(
    aux: Dict[str, torch.Tensor],
    cfg: Union[MoEConfig, BenchConfig],
    ref: torch.Tensor,
) -> torch.Tensor:
  """نسخهٔ امن وقتی aux خالی است."""
  if isinstance(cfg, BenchConfig):
    cfg = cfg.moe
  s = ref.new_tensor(0.0)
  if "load_balance" in aux and cfg.use_load_balance:
    s = s + aux["load_balance"] * cfg.load_balance_coef
  if "z" in aux and cfg.use_z_loss:
    s = s + aux["z"] * cfg.z_loss_coef
  if "router_entropy" in aux and cfg.use_router_entropy_aux:
    s = s + aux["router_entropy"] * cfg.router_entropy_coef
  return s


# =============================================================================
# 4) Dispatch مرتب‌شده (هر expert حداکل یک بار forward روی زیرمجموعه)
# =============================================================================


def _capacity_per_expert(
    batch: int, top_k: int, num_experts: int, factor: float
) -> int:
  return max(1, int(math.ceil(factor * batch * top_k / float(num_experts))))


def _subsample_layer_tensors(
    h: torch.Tensor,
    upstream: Optional[torch.Tensor],
    max_tokens: int,
    topk_idx: Optional[torch.Tensor] = None,
    router_logits: Optional[torch.Tensor] = None,
    expert_outputs: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
  b = int(h.shape[0])
  if b <= max_tokens:
    return h, upstream, topk_idx, router_logits, expert_outputs
  idx = torch.randperm(b, device=h.device)[:max_tokens]
  h_s = h.index_select(0, idx)
  upstream_s = upstream.index_select(0, idx) if upstream is not None else None
  topk_s = topk_idx.index_select(0, idx) if topk_idx is not None else None
  router_s = router_logits.index_select(0, idx) if router_logits is not None else None
  expert_s = (
      expert_outputs.index_select(0, idx) if expert_outputs is not None else None
  )
  return h_s, upstream_s, topk_s, router_s, expert_s


def _expert_grad_direction(
    expert: nn.Module,
    z: torch.Tensor,
    upstream: torch.Tensor,
) -> Optional[torch.Tensor]:
  out = expert(z)
  scalar = (out * upstream).sum()
  params = [p for p in expert.parameters() if p.requires_grad]
  if not params:
    return None
  grads = torch.autograd.grad(
      scalar,
      params,
      create_graph=True,
      retain_graph=True,
      allow_unused=True,
  )
  parts = [g.reshape(-1) for g in grads if g is not None]
  if not parts:
    return None
  flat = torch.cat(parts)
  if float(flat.norm().item()) < 1e-12:
    return None
  return F.normalize(flat, dim=0, eps=1e-8)


def grad_direction_diversity_loss(
    layer: "SparseMoELayer",
    h: torch.Tensor,
    upstream: torch.Tensor,
    topk_idx: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, float]:
  h, upstream_opt, topk_opt, _, _ = _subsample_layer_tensors(
      h, upstream, max_tokens, topk_idx=topk_idx
  )
  assert upstream_opt is not None and topk_opt is not None
  upstream = upstream_opt
  topk_idx = topk_opt
  z = layer.norm(h)
  # GDDS را محلی نگه می‌داریم؛ router با router_div_loss سبک‌تر آموزش می‌بیند.
  upstream = upstream.detach()

  loss = z.new_tensor(0.0)
  cos_abs_sum = 0.0
  n_pairs = 0
  n_experts = len(layer.experts)
  for i in range(n_experts):
    active_i = (topk_idx == i).any(dim=1)
    for j in range(i + 1, n_experts):
      # Masked GDDS: فقط توکن‌هایی که هر دو expert در top-k همان توکن هستند.
      active = active_i & (topk_idx == j).any(dim=1)
      if not bool(active.any().item()):
        continue
      z_pair = z[active]
      upstream_pair = upstream[active]
      dir_i = _expert_grad_direction(layer.experts[i], z_pair, upstream_pair)
      dir_j = _expert_grad_direction(layer.experts[j], z_pair, upstream_pair)
      if dir_i is None or dir_j is None:
        continue
      cos = (dir_i * dir_j).sum()
      loss = loss + cos * cos
      cos_abs_sum += float(cos.detach().abs().item())
      n_pairs += 1

  if n_pairs < 1:
    return z.new_tensor(0.0), 0.0

  loss = loss / float(max(n_pairs, 1))
  return loss, cos_abs_sum / float(max(n_pairs, 1))


def router_output_diversity_loss(
    layer: "SparseMoELayer",
    h: torch.Tensor,
    router_logits: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, float, float]:
  h, _, _, router_logits_opt, _ = _subsample_layer_tensors(
      h, None, max_tokens, router_logits=router_logits
  )
  assert router_logits_opt is not None
  router_logits = router_logits_opt
  z = layer.norm(h)
  probs = F.softmax(router_logits, dim=-1)
  expert_outs = torch.stack([expert(z) for expert in layer.experts], dim=1)
  expert_outs = F.normalize(expert_outs, dim=-1, eps=1e-8)
  sim = torch.einsum("bed,bfd->bef", expert_outs, expert_outs).pow(2).detach()
  pair_prob = probs.unsqueeze(2) * probs.unsqueeze(1)
  e = int(probs.shape[-1])
  pair_mask = torch.triu(
      torch.ones(e, e, device=h.device, dtype=torch.bool), diagonal=1
  )
  weighted = sim[:, pair_mask] * pair_prob[:, pair_mask]
  loss = weighted.mean()
  mean_sim = float(sim[:, pair_mask].mean().detach().item())
  mean_pair_prob = float(pair_prob[:, pair_mask].mean().detach().item())
  return loss, mean_sim, mean_pair_prob


def router_grad_diversity_loss(
    expert_outputs: torch.Tensor,
    upstream: torch.Tensor,
    router_logits: torch.Tensor,
    topk_idx: torch.Tensor,
    max_tokens: int,
    *,
    detach_upstream: bool = True,
    detach_sim: bool = True,
) -> Tuple[torch.Tensor, float, float]:
  """GRD: جریمه router وقتی expertهای فعال learning response مشابه دارند.

  R_e = normalize(o_e) ⊙ normalize(upstream_proxy)
  L = mean_{i<j, both in top-k} cos(R_i, R_j)^2 · p_i · p_j
  """
  b = int(expert_outputs.shape[0])
  if b > max_tokens:
    idx = torch.randperm(b, device=expert_outputs.device)[:max_tokens]
    upstream = upstream.index_select(0, idx)
    topk_idx = topk_idx.index_select(0, idx)
    router_logits = router_logits.index_select(0, idx)
    expert_outputs = expert_outputs.index_select(0, idx)

  upstream = upstream.detach() if detach_upstream else upstream
  o_norm = F.normalize(expert_outputs, dim=-1, eps=1e-8)
  u_norm = F.normalize(upstream, dim=-1, eps=1e-8).unsqueeze(1)
  responses = F.normalize(o_norm * u_norm, dim=-1, eps=1e-8)
  probs = F.softmax(router_logits, dim=-1)

  loss = expert_outputs.new_tensor(0.0)
  n_terms = 0
  cos_abs_sum = 0.0
  pair_prob_sum = 0.0
  n_experts = int(expert_outputs.shape[1])
  for i in range(n_experts):
    active_i = (topk_idx == i).any(dim=1)
    for j in range(i + 1, n_experts):
      active = active_i & (topk_idx == j).any(dim=1)
      if not bool(active.any().item()):
        continue
      ri = responses[active, i]
      rj = responses[active, j]
      cos = (ri * rj).sum(dim=-1)
      sim = cos * cos
      if detach_sim:
        sim = sim.detach()
      pi = probs[active, i]
      pj = probs[active, j]
      term = (sim * pi * pj).mean()
      loss = loss + term
      cos_abs_sum += float(cos.detach().abs().mean().item())
      pair_prob_sum += float((pi * pj).mean().detach().item())
      n_terms += 1

  if n_terms < 1:
    return expert_outputs.new_tensor(0.0), 0.0, 0.0

  loss = loss / float(n_terms)
  return loss, cos_abs_sum / float(n_terms), pair_prob_sum / float(n_terms)


def _router_grad_div_layer_indices(
    model: "CifarGDDSMoEBackbone", cfg: BenchConfig
) -> List[int]:
  if not model.cfg.use_hybrid_spatial_moe:
    return list(range(len(model.moe_layers)))
  if not cfg.router_grad_div_global_only:
    return list(range(len(model.moe_layers)))
  n_sp = len(model.spatial_moe_layers)
  return list(range(n_sp, len(model.moe_layers)))


def compute_diversity_losses(
    model: "CifarGDDSMoEBackbone",
    trace: Dict[str, List[torch.Tensor]],
    task_loss: torch.Tensor,
    cfg: BenchConfig,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
  """GDDS (expert) + GRD (router) با یک بار upstream per layer."""
  first = trace["inputs"][0]
  grad_total = first.new_tensor(0.0)
  grd_total = first.new_tensor(0.0)
  stats: Dict[str, float] = {}

  grad_layers = (
      set(_grad_div_layer_indices(model, cfg)) if cfg.use_grad_div else set()
  )
  grd_layers = (
      set(_router_grad_div_layer_indices(model, cfg))
      if cfg.use_router_grad_div
      else set()
  )
  for li in sorted(grad_layers | grd_layers):
    layer = model.moe_layers[li]
    y_li = trace["outputs"][li]
    prefix = model.layer_prefix(li)
    need_gdds = li in grad_layers
    need_grd = li in grd_layers

    create_graph = bool(need_gdds)
    upstream = torch.autograd.grad(
        task_loss,
        y_li,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )[0]
    if upstream is None:
      continue

    if need_gdds:
      max_tok = max(int(cfg.grad_div_max_tokens), 1)
      div_li, mean_abs_cos = grad_direction_diversity_loss(
          layer,
          trace["inputs"][li],
          upstream,
          trace["topk_indices"][li],
          max_tok,
      )
      grad_total = grad_total + div_li
      stats[f"{prefix}_grad_div"] = float(div_li.detach().item())
      stats[f"{prefix}_grad_cos_abs"] = float(mean_abs_cos)

    if need_grd:
      expert_outs = trace["expert_outputs"][li]
      max_tok = max(int(cfg.router_grad_div_max_tokens), 1)
      grd_li, mean_resp_cos, mean_pair_prob = router_grad_diversity_loss(
          expert_outs,
          upstream,
          trace["router_logits"][li],
          trace["topk_indices"][li],
          max_tok,
          detach_upstream=cfg.router_grad_div_detach_upstream,
          detach_sim=cfg.router_grad_div_detach_sim,
      )
      grd_total = grd_total + grd_li
      stats[f"{prefix}_router_grd"] = float(grd_li.detach().item())
      stats[f"{prefix}_router_grd_resp_cos"] = float(mean_resp_cos)
      stats[f"{prefix}_router_grd_pair_prob"] = float(mean_pair_prob)

  if any(k.endswith("_grad_div") for k in stats):
    stats["grad_div_total"] = float(grad_total.detach().item())
  if any(k.endswith("_router_grd") for k in stats):
    stats["router_grd_total"] = float(grd_total.detach().item())
  return grad_total, grd_total, stats


def _grad_div_layer_indices(model: "CifarGDDSMoEBackbone", cfg: BenchConfig) -> List[int]:
  if not model.cfg.use_hybrid_spatial_moe:
    return list(range(len(model.moe_layers)))
  use_global = bool(cfg.grad_div_global_only)
  use_spatial = bool(cfg.grad_div_spatial_only)
  if not use_global and not use_spatial:
    return list(range(len(model.moe_layers)))
  n_sp = len(model.spatial_moe_layers)
  out: List[int] = []
  if use_spatial:
    out.extend(range(n_sp))
  if use_global:
    out.extend(range(n_sp, len(model.moe_layers)))
  return out


def compute_grad_div_losses(
    model: "CifarGDDSMoEBackbone",
    trace: Dict[str, List[torch.Tensor]],
    task_loss: torch.Tensor,
    cfg: BenchConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
  total = task_loss.new_tensor(0.0)
  stats: Dict[str, float] = {}
  if not cfg.use_grad_div:
    return total, stats

  max_tok = max(int(cfg.grad_div_max_tokens), 1)
  for li in _grad_div_layer_indices(model, cfg):
    layer = model.moe_layers[li]
    y_li = trace["outputs"][li]
    h_li = trace["inputs"][li]
    topk_li = trace["topk_indices"][li]
    upstream = torch.autograd.grad(
        task_loss,
        y_li,
        retain_graph=True,
        create_graph=True,
        allow_unused=True,
    )[0]
    if upstream is None:
      continue
    div_li, mean_abs_cos = grad_direction_diversity_loss(
        layer, h_li, upstream, topk_li, max_tok
    )
    total = total + div_li
    prefix = model.layer_prefix(li)
    stats[f"{prefix}_grad_div"] = float(div_li.detach().item())
    stats[f"{prefix}_grad_cos_abs"] = float(mean_abs_cos)
  if stats:
    stats["grad_div_total"] = float(total.detach().item())
  return total, stats


def compute_router_div_losses(
    model: "CifarGDDSMoEBackbone",
    trace: Dict[str, List[torch.Tensor]],
    cfg: BenchConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
  first = trace["inputs"][0]
  total = first.new_tensor(0.0)
  stats: Dict[str, float] = {}
  if not cfg.use_router_div:
    return total, stats
  max_tok = max(int(cfg.router_div_max_tokens), 1)
  for li, layer in enumerate(model.moe_layers):
    h_li = trace["inputs"][li]
    logits_li = trace["router_logits"][li]
    div_li, mean_sim, mean_pair_prob = router_output_diversity_loss(
        layer, h_li, logits_li, max_tok
    )
    total = total + div_li
    prefix = model.layer_prefix(li)
    stats[f"{prefix}_router_div"] = float(div_li.detach().item())
    stats[f"{prefix}_router_div_sim"] = float(mean_sim)
    stats[f"{prefix}_router_div_pair_prob"] = float(mean_pair_prob)
  if stats:
    stats["router_div_total"] = float(total.detach().item())
  return total, stats


def _deterministic_topk(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
  """Top-k با شکستن تساوی پایدار: برای logits برابر، ایندکس کوچک‌تر اولویت دارد."""
  if dim != -1:
    x = x.movedim(dim, -1)
  perm = torch.argsort(x, dim=-1, stable=True, descending=True)
  return perm[..., :k]


def sparse_dispatch_sorted(
    h: torch.Tensor,
    topk_idx: torch.Tensor,
    combine: torch.Tensor,
    experts: nn.ModuleList,
    top_k: int,
    capacity_per_expert: Optional[int],
    num_experts: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
  """
  flat کردن (B,K) → sort بر اساس expert → برای هر expert یکبار forward.
  اگر capacity_per_expert ست باشد، داخل هر گروه با وزن combine برش می‌خورد.
  """
  B, dim = h.shape
  device, dtype = h.device, h.dtype
  flat_e = topk_idx.reshape(-1)
  flat_t = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(top_k)
  flat_w = combine.reshape(-1)
  moe_out = torch.zeros(B, dim, device=device, dtype=dtype)
  n_flat = int(flat_e.numel())
  desired = torch.zeros(num_experts, device=device, dtype=torch.long)
  kept = torch.zeros(num_experts, device=device, dtype=torch.long)
  stats: Dict[str, Any] = {
      "n_flat": n_flat,
      "dropped_assignments": 0,
      "expert_calls": 0,
      "capacity_per_expert": capacity_per_expert,
  }
  if n_flat == 0:
    z = [0] * num_experts
    stats.update(
        {
            "n_kept_assignments": 0,
            "drop_rate": 0.0,
            "capacity_usage_ratio": 0.0,
            "mean_tokens_per_expert_fwd": 0.0,
            "std_tokens_per_expert_fwd": 0.0,
            "active_experts_ratio_fwd": 0.0,
            "tokens_per_expert_kept": z,
            "tokens_per_expert_desired": z,
        }
    )
    return moe_out, stats

  order = flat_e.argsort(stable=True)
  fe = flat_e[order]
  ft = flat_t[order]
  fw = flat_w[order]
  uniq, counts = fe.unique_consecutive(return_counts=True)
  ptr = 0
  for ue, cnt in zip(uniq.tolist(), counts.tolist()):
    cnt = int(cnt)
    ue_i = int(ue)
    desired[ue_i] += cnt
    seg_t = ft[ptr : ptr + cnt]
    seg_w = fw[ptr : ptr + cnt]
    dropped_here = 0
    if capacity_per_expert is not None and cnt > capacity_per_expert:
      sub = torch.argsort(seg_w, descending=True, stable=True)
      keep = sub[:capacity_per_expert]
      seg_t = seg_t[keep]
      seg_w = seg_w[keep]
      dropped_here = cnt - seg_t.numel()
    if seg_t.numel() == 0:
      ptr += cnt
      stats["dropped_assignments"] += dropped_here
      continue

    # renormalize for stability
    seg_w = seg_w / (seg_w.sum() + 1e-9)

    kept[ue_i] += int(seg_t.numel())
    h_sub = h.index_select(0, seg_t)
    out_sub = experts[int(ue)](h_sub)
    w = seg_w.unsqueeze(-1)
    moe_out.index_add_(0, seg_t, out_sub * w)
    stats["expert_calls"] += 1
    stats["dropped_assignments"] += dropped_here
    ptr += cnt
  dropped = int(stats["dropped_assignments"])
  nk = n_flat - dropped
  stats["n_kept_assignments"] = nk
  stats["drop_rate"] = float(dropped) / float(n_flat)
  stats["capacity_usage_ratio"] = float(nk) / float(n_flat)
  kf = kept.float()
  stats["mean_tokens_per_expert_fwd"] = float(kf.mean().item())
  stats["std_tokens_per_expert_fwd"] = float(kf.std(unbiased=False).item())
  stats["active_experts_ratio_fwd"] = float((kept > 0).float().mean().item())
  stats["tokens_per_expert_kept"] = kept.cpu().tolist()
  stats["tokens_per_expert_desired"] = desired.cpu().tolist()
  return moe_out, stats


# =============================================================================
# 5) لایهٔ Sparse MoE
# =============================================================================


class SparseMoELayer(nn.Module):
  def __init__(self, dim: int, cfg: Union[MoEConfig, BenchConfig]) -> None:
    super().__init__()
    # Guardrail: اگر اشتباهاً BenchConfig پاس شد، MoEConfig داخلی را بردار.
    if isinstance(cfg, BenchConfig):
      cfg = cfg.moe
    self.cfg = cfg
    if cfg.top_k > cfg.num_experts or cfg.top_k < 1:
      raise ValueError("top_k must be in [1, num_experts]")
    self.norm = nn.LayerNorm(dim)
    self.router = nn.Linear(dim, cfg.num_experts, bias=True)
    nn.init.normal_(self.router.weight, mean=0.0, std=0.02)
    self.router.bias.data.uniform_(-1e-3, 1e-3)
    self.experts = nn.ModuleList(
        [
            ExpertMLP(dim, cfg.expert_hidden, cfg.expert_dropout)
            for _ in range(cfg.num_experts)
        ]
    )

  def forward(
      self,
      x: torch.Tensor,
      mask_expert_id: Optional[int] = None,
      return_expert_outputs: bool = False,
  ) -> Tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      Dict[str, torch.Tensor],
      Dict[str, Any],
      Optional[torch.Tensor],
  ]:
    h = self.norm(x)
    router_logits = self.router(h) # [B, num_experts]
    t = max(self.cfg.router_temperature, 1e-6)
    scaled = router_logits / t # [0.2, 1.5, 0.3, 2.0]
    if mask_expert_id is not None:
      scaled = scaled.clone()
      scaled[:, mask_expert_id] = -1e9
    if self.training and self.cfg.router_noise_std > 0.0:
      scaled = scaled + torch.randn_like(scaled) * self.cfg.router_noise_std
    src = scaled.detach() if self.cfg.router_detach_for_topk else scaled
    topk_idx = _deterministic_topk(src, self.cfg.top_k, dim=-1)
    topk_val = scaled.gather(1, topk_idx) #[2.0, 1.5]
    if self.cfg.combine_method == "softmax":
      combine = F.softmax(topk_val, dim=-1) #[0.62, 0.38]
    else:
      combine = F.softmax(topk_val, dim=-1)

    cap = None
    if self.cfg.capacity_factor is not None:
      cap = _capacity_per_expert(
          h.shape[0], self.cfg.top_k, self.cfg.num_experts, self.cfg.capacity_factor
      )

    expert_outputs: Optional[torch.Tensor] = None
    if return_expert_outputs:
      expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=1)

    # h: ویژگی توکن‌ها
    # topk_idx: کدوم expert
    # combine: وزن هر expert
    # experts: خود MLPها
    moe_out, d_stats = sparse_dispatch_sorted(
        h, topk_idx, combine, self.experts, self.cfg.top_k, cap, self.cfg.num_experts
    )
    # moe_out: [B, dim]
    # moe_out = خروجی ترکیب‌شدهٔ expertها برای هر توکن
    y = x + self.cfg.moe_residual_scale * moe_out
    aux = build_aux_losses(router_logits, topk_idx, self.cfg, routing_logits=scaled)
    return y, router_logits, scaled, topk_idx, aux, d_stats, expert_outputs


# =============================================================================
# 6) Backbone Tiny ImageNet-200 (ResNet-lite, stage3 عریض‌تر + عمیق‌تر)
# =============================================================================

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
    super().__init__()

    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(out_ch)

    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(out_ch)

    self.act = nn.GELU()

    self.shortcut = nn.Identity()
    if stride != 1 or in_ch != out_ch:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
          nn.BatchNorm2d(out_ch),
      )



  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.act(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out = out + self.shortcut(x)
    out = self.act(out)
    return out


class CifarGDDSMoEBackbone(nn.Module):
  def __init__(self, cfg: BenchConfig) -> None:
    super().__init__()

    self.cfg = cfg

    self.stem = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.GELU(),
    )

    self.stage1 = nn.Sequential(
        BasicBlock(64, 64),
        BasicBlock(64, 64),
    )

    self.stage2 = nn.Sequential(
        BasicBlock(64, 128, stride=2),
        BasicBlock(128, 128),
    )

    fc = int(cfg.feature_channels)
    s3_depth = max(int(cfg.backbone_stage3_depth), 1)
    s3_blocks: List[BasicBlock] = [BasicBlock(128, fc, stride=2)]
    s3_blocks.extend(BasicBlock(fc, fc) for _ in range(s3_depth - 1))
    self.stage3 = nn.Sequential(*s3_blocks)

    self.pool = nn.AdaptiveAvgPool2d(1)

    self.embed = nn.Linear(cfg.feature_channels, cfg.dim)

    if cfg.use_hybrid_spatial_moe:
      spatial_moe, global_moe = resolve_hybrid_moe_configs(cfg)
      self.spatial_moe_layers = nn.ModuleList(
          [
              SparseMoELayer(cfg.dim, spatial_moe)
              for _ in range(cfg.moe_num_spatial_layers)
          ]
      )
      self.global_moe_layers = nn.ModuleList(
          [
              SparseMoELayer(cfg.dim, global_moe)
              for _ in range(cfg.moe_num_global_layers)
          ]
      )
      self.moe_layers = nn.ModuleList(
          list(self.spatial_moe_layers) + list(self.global_moe_layers)
      )
      self._moe_layer_cfgs: List[MoEConfig] = (
          [spatial_moe] * cfg.moe_num_spatial_layers
          + [global_moe] * cfg.moe_num_global_layers
      )
      n_tokens = self._infer_spatial_tokens()
      if cfg.use_spatial_pos_embed:
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, n_tokens, cfg.dim))
      else:
        self.register_parameter("spatial_pos_embed", None)
    else:
      self.spatial_moe_layers = nn.ModuleList()
      self.global_moe_layers = nn.ModuleList()
      self.moe_layers = nn.ModuleList(
          [SparseMoELayer(cfg.dim, cfg.moe) for _ in range(cfg.moe_num_layers)]
      )
      self._moe_layer_cfgs = [cfg.moe] * cfg.moe_num_layers
      self._spatial_tokens = 0
      self.register_parameter("spatial_pos_embed", None)

    self.final_norm = nn.LayerNorm(cfg.dim)
    self.head = nn.Linear(cfg.dim, cfg.num_classes)

    nn.init.xavier_uniform_(self.embed.weight)
    nn.init.zeros_(self.embed.bias)

    nn.init.xavier_uniform_(self.head.weight, gain=1.0)
    nn.init.zeros_(self.head.bias)

    self._init_conv_weights()

  def _init_conv_weights(self) -> None:
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

  def _infer_spatial_tokens(self) -> int:
    with torch.no_grad():
      feat = self.encode_features(
          torch.zeros(1, 3, self.cfg.image_size, self.cfg.image_size)
      )
    self._spatial_hw = (int(feat.shape[2]), int(feat.shape[3]))
    n_tokens = int(feat.shape[2] * feat.shape[3])
    self._spatial_tokens = n_tokens
    return n_tokens

  def encode_features(self, x: torch.Tensor) -> torch.Tensor:
    x = self.stem(x)
    x = self.stage1(x)
    x = self.stage2(x)
    return self.stage3(x)

  def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
    """[B, T, dim] — توکن‌های spatial قبل از MoE."""
    feat = self.encode_features(x)
    b, _c, h, w = feat.shape
    tokens = feat.permute(0, 2, 3, 1).reshape(b, h * w, self.cfg.feature_channels)
    return self.embed(tokens)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """ورودی لایهٔ اول MoE: [B, dim] (global) یا mean توکن spatial (بدون MoE)."""
    if self.cfg.use_hybrid_spatial_moe:
      return self.encode_spatial(x).mean(dim=1)
    feat = self.encode_features(x)
    pooled = self.pool(feat).flatten(1)
    return self.embed(pooled)

  def moe_config_for_layer(self, li: int) -> MoEConfig:
    return self._moe_layer_cfgs[li]

  def num_experts_for_layer(self, li: int) -> int:
    return int(self._moe_layer_cfgs[li].num_experts)

  def layer_prefix(self, li: int) -> str:
    if not self.cfg.use_hybrid_spatial_moe:
      return f"L{li}"
    n_sp = len(self.spatial_moe_layers)
    if li < n_sp:
      return f"S{li}"
    return f"G{li - n_sp}"

  def forward(
      self,
      x: torch.Tensor,
      return_router: bool = False,
      return_moe_trace: bool = False,
      mask_expert: Optional[Tuple[int, int]] = None,
  ):
    aux_list: List[Dict[str, torch.Tensor]] = []
    stats_list: List[Dict[str, Any]] = []
    routing: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    moe_inputs: List[torch.Tensor] = []
    moe_outputs: List[torch.Tensor] = []
    moe_router_logits: List[torch.Tensor] = []
    moe_topk_indices: List[torch.Tensor] = []
    moe_expert_outputs: List[torch.Tensor] = []
    need_expert_outs = bool(return_moe_trace and self.cfg.use_router_grad_div)

    if self.cfg.use_hybrid_spatial_moe:
      tokens = self.encode_spatial(x)
      if self.spatial_pos_embed is not None:
        tokens = tokens + self.spatial_pos_embed
      b, t, d = tokens.shape
      h = tokens.reshape(b * t, d)
      n_sp = len(self.spatial_moe_layers)
    else:
      h = self.encode(x)
      n_sp = 0

    for li, layer in enumerate(self.moe_layers):
      if self.cfg.use_hybrid_spatial_moe and li == n_sp:
        h = h.reshape(b, t, d).mean(dim=1)
      mask_id: Optional[int] = None
      if mask_expert is not None and mask_expert[0] == li:
        mask_id = mask_expert[1]
      if return_moe_trace:
        moe_inputs.append(h)
      h, rlog, scaled, tidx, aux, st, eout = layer(
          h,
          mask_expert_id=mask_id,
          return_expert_outputs=need_expert_outs,
      )
      if return_moe_trace:
        moe_outputs.append(h)
        moe_router_logits.append(scaled)
        moe_topk_indices.append(tidx)
        if need_expert_outs:
          assert eout is not None
          moe_expert_outputs.append(eout)
      aux_list.append(aux)
      stats_list.append(st)
      if return_router:
        routing.append((rlog, scaled, tidx))

    h = self.final_norm(h)
    logits = self.head(h)

    trace = {
        "inputs": moe_inputs,
        "outputs": moe_outputs,
        "router_logits": moe_router_logits,
        "topk_indices": moe_topk_indices,
        "expert_outputs": moe_expert_outputs,
    }
    if return_router and return_moe_trace:
      return logits, aux_list, stats_list, routing, trace
    if return_router:
      return logits, aux_list, stats_list, routing
    if return_moe_trace:
      return logits, aux_list, stats_list, trace

    return logits, aux_list, stats_list


CifarSparseMoEBackbone = CifarGDDSMoEBackbone


# =============================================================================
# 7) دادهٔ Tiny ImageNet-200 (لوکال سبک)
# =============================================================================

TINY_IMAGENET200_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _tiny_imagenet200_ok(base: str) -> bool:
  return (
      os.path.isfile(os.path.join(base, "wnids.txt"))
      and os.path.isdir(os.path.join(base, "train"))
      and os.path.isfile(os.path.join(base, "val", "val_annotations.txt"))
      and os.path.isdir(os.path.join(base, "val", "images"))
  )


def _extract_tiny_imagenet_archive(archive: str, root: str) -> None:
  try:
    with zipfile.ZipFile(archive, "r") as zf:
      zf.extractall(path=root)
  except (OSError, zipfile.BadZipFile):
    pass


def ensure_tiny_imagenet200(root: str, local: Optional[str] = None) -> str:
  base = os.path.join(root, "tiny-imagenet-200")
  if _tiny_imagenet200_ok(base):
    return base
  os.makedirs(root, exist_ok=True)
  u = (
      os.path.abspath(os.path.expanduser(str(local).strip().strip('"')))
      if local and str(local).strip()
      else None
  )
  if u and os.path.isdir(u) and _tiny_imagenet200_ok(u):
    dst = base
    if not os.path.exists(dst):
      shutil.copytree(u, dst)
    return base
  if u and os.path.isfile(u):
    _extract_tiny_imagenet_archive(u, root)
  for name in ("tiny-imagenet-200.zip",):
    p = os.path.join(root, name)
    if os.path.isfile(p):
      _extract_tiny_imagenet_archive(p, root)
    if _tiny_imagenet200_ok(base):
      return base
  zip_path = os.path.join(root, "tiny-imagenet-200.zip")
  if not os.path.isfile(zip_path):
    print(f"[data] downloading Tiny ImageNet-200 → {zip_path}", flush=True)
    urllib.request.urlretrieve(TINY_IMAGENET200_URL, zip_path)
  _extract_tiny_imagenet_archive(zip_path, root)
  if not _tiny_imagenet200_ok(base):
    raise RuntimeError(f"Tiny ImageNet-200 not found under {base}")
  return base


class TinyImageNet200Dataset(Dataset):
  """train / val splits for Tiny ImageNet-200 (200 classes, 64×64)."""

  def __init__(
      self,
      root: str,
      split: str = "train",
      transform: Optional[T.Compose] = None,
  ) -> None:
    base = os.path.join(root, "tiny-imagenet-200")
    if not _tiny_imagenet200_ok(base):
      raise FileNotFoundError(f"Tiny ImageNet-200 missing: {base}")
    self.transform = transform
    self.samples: List[Tuple[str, int]] = []
    with open(os.path.join(base, "wnids.txt"), encoding="utf-8") as f:
      wnids = [line.strip() for line in f if line.strip()]
    class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
    split = split.lower()
    if split == "train":
      for wnid in wnids:
        img_dir = os.path.join(base, "train", wnid, "images")
        for fname in sorted(os.listdir(img_dir)):
          if fname.lower().endswith((".jpeg", ".jpg", ".png")):
            self.samples.append(
                (os.path.join(img_dir, fname), class_to_idx[wnid])
            )
    elif split in ("val", "valid", "validation"):
      ann = os.path.join(base, "val", "val_annotations.txt")
      img_dir = os.path.join(base, "val", "images")
      with open(ann, encoding="utf-8") as f:
        for line in f:
          parts = line.strip().split("\t")
          if len(parts) < 2:
            continue
          fname, wnid = parts[0], parts[1]
          if wnid in class_to_idx:
            self.samples.append(
                (os.path.join(img_dir, fname), class_to_idx[wnid])
            )
    else:
      raise ValueError(f"unsupported split: {split!r}")
    if not self.samples:
      raise RuntimeError(f"no samples for split={split!r} under {base}")

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(self, index: int):
    from PIL import Image

    path, label = self.samples[index]
    img = Image.open(path).convert("RGB")
    if self.transform is not None:
      img = self.transform(img)
    return img, label


def _seed_torchvision_random(seed: int) -> None:
  """RandAugment / RandomCrop از random پایتون و torch هر دو استفاده می‌کنند."""
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)


def loaders(
    cfg: BenchConfig,
) -> Tuple[DataLoader, DataLoader]:
  set_reproducibility(cfg.seed)
  _seed_torchvision_random(cfg.seed)
  ensure_tiny_imagenet200(cfg.data_dir, cfg.local_tiny_imagenet)
  sz = int(cfg.image_size)
  mean = (0.485, 0.456, 0.406)
  std = (0.229, 0.224, 0.225)
  train_tfm_list: List[nn.Module] = []
  if cfg.train_random_crop_pad > 0:
    train_tfm_list.append(T.RandomCrop(sz, padding=cfg.train_random_crop_pad))
  if cfg.train_hflip:
    train_tfm_list.append(T.RandomHorizontalFlip())
  if cfg.train_randaugment_ops > 0:
    train_tfm_list.append(
        T.RandAugment(
            num_ops=cfg.train_randaugment_ops,
            magnitude=cfg.train_randaugment_magnitude,
        )
    )
  train_tfm_list.extend([T.ToTensor(), T.Normalize(mean, std)])
  train_tfm = T.Compose(train_tfm_list)
  test_tfm = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
  g_train = torch.Generator().manual_seed(cfg.seed)
  g_test = torch.Generator().manual_seed(cfg.seed + 10_000)
  w_init = _dataloader_seed_worker if cfg.num_workers > 0 else None
  tr = TinyImageNet200Dataset(cfg.data_dir, "train", transform=train_tfm)
  te = TinyImageNet200Dataset(cfg.data_dir, "val", transform=test_tfm)
  pin = torch.cuda.is_available()
  train = DataLoader(
      tr,
      cfg.batch_size,
      shuffle=True,
      num_workers=cfg.num_workers,
      pin_memory=False,
      generator=g_train,
      worker_init_fn=w_init,
  )
  test = DataLoader(
      te,
      cfg.batch_size,
      shuffle=False,
      num_workers=cfg.num_workers,
      pin_memory=False,
      generator=g_test,
      worker_init_fn=w_init,
  )
  return train, test


# =============================================================================
# 7.5) Expert specialization metrics (۵ متریک اصلی)
# =============================================================================


def _accumulate_routing_class_counts(
    counts: torch.Tensor,
    class_totals: torch.Tensor,
    labels: torch.Tensor,
    topk_idx: torch.Tensor,
    update_class_totals: bool = True,
) -> None:
  """جمع‌آوری تعداد انتخاب expert e برای نمونه‌های کلاس c."""
  b = labels.shape[0]
  c = labels.long()
  num_classes = int(class_totals.shape[0])
  for k in range(topk_idx.shape[1]):
    e = topk_idx[:, k].long()
    idx = e * num_classes + c
    counts.view(-1).index_add_(
        0, idx, torch.ones(b, device=counts.device, dtype=counts.dtype)
    )
  if update_class_totals:
    for cls in range(num_classes):
      class_totals[cls] += float((c == cls).sum().item())


def _routing_affinity_matrix(
    counts: torch.Tensor,
    class_totals: torch.Tensor,
) -> torch.Tensor:
  """M[e,c] = P(expert e در top-k | کلاس c)."""
  denom = class_totals.clamp_min(1.0).unsqueeze(0)
  return counts.float() / denom


def _entropy_from_probs(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
  p = p.clamp_min(1e-12)
  return -(p * p.log()).sum(dim=dim)


def _expert_entropy_metrics(affinity: torch.Tensor) -> Dict[str, torch.Tensor]:
  """از M[e,c]: entropy، N_eff و purity هر expert."""
  row_sum = affinity.sum(dim=1, keepdim=True).clamp_min(1e-12)
  p_ec = affinity / row_sum
  ent = _entropy_from_probs(p_ec, dim=1)
  n_eff = torch.exp(ent)
  purity = p_ec.max(dim=1).values
  return {
      "expert_entropy": ent,
      "expert_effective_num_classes": n_eff,
      "expert_routing_purity": purity,
      "mean_expert_entropy": ent.mean(),
      "mean_effective_num_classes": n_eff.mean(),
      "mean_routing_purity": purity.mean(),
  }


@torch.no_grad()
def _expert_output_similarity(moe_layer: SparseMoELayer, h: torch.Tensor) -> float:
  """میانگین cosine similarity بین جفت expertها (off-diagonal)."""
  z = moe_layer.norm(h)
  outs = torch.stack([expert(z) for expert in moe_layer.experts], dim=0)
  outs = F.normalize(outs, dim=-1)
  e = outs.shape[0]
  sim = torch.einsum("ebd,fbd->efb", outs, outs).mean(dim=-1)
  mask = ~torch.eye(e, dtype=torch.bool, device=sim.device)
  return float(sim[mask].mean().item())


@torch.no_grad()
def _expert_output_effective_rank(moe_layer: SparseMoELayer, h: torch.Tensor) -> float:
  """effective rank از covariance خروجی expertها."""
  z = moe_layer.norm(h)
  outs = torch.stack([expert(z) for expert in moe_layer.experts], dim=0)
  y = outs.permute(1, 0, 2).reshape(-1, outs.shape[-1])
  y = y - y.mean(dim=0, keepdim=True)
  n = max(y.shape[0] - 1, 1)
  # eigvalsh روی CPU: cusolver در برخی Colab/CUDA setupها خطا می‌دهد.
  yc = y.detach().cpu()
  cov = (yc.T @ yc) / float(n)
  eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
  s1 = eigvals.sum()
  s2 = eigvals.pow(2).sum()
  if float(s2.item()) < 1e-12:
    return 0.0
  return float((s1 * s1 / s2).item())


def _pca_spread(coords: np.ndarray) -> float:
  if coords.shape[0] < 2:
    return 0.0
  diffs = coords[:, None, :] - coords[None, :, :]
  dists = np.sqrt((diffs ** 2).sum(-1))
  mask = ~np.eye(coords.shape[0], dtype=bool)
  return float(dists[mask].mean())


@torch.no_grad()
def _expert_cka_pca_metrics(moe_layer: SparseMoELayer, h: torch.Tensor) -> Dict[str, Any]:
  """linear CKA خروجی expertها + PCA وزن expert/router (مثل expert_analysis)."""
  z = moe_layer.norm(h)
  outs = torch.stack([expert(z) for expert in moe_layer.experts], dim=0)
  e = int(outs.shape[0])
  cka_mat = np.zeros((e, e), dtype=np.float64)
  for i in range(e):
    for j in range(e):
      cka_mat[i, j] = _linear_cka(outs[i], outs[j])
  offdiag = cka_mat[~np.eye(e, dtype=bool)] if e > 1 else np.array([], dtype=np.float64)
  wvec = _expert_weight_vectors(moe_layer)
  weight_pca2 = _pca_2d(wvec)
  router_w = moe_layer.router.weight.detach().cpu().numpy()
  router_pca2 = _pca_2d(router_w)
  return {
      "expert_linear_cka": cka_mat.round(4).tolist(),
      "expert_output_cka_offdiag": float(offdiag.mean()) if offdiag.size else 0.0,
      "weight_pca_2d": weight_pca2.round(4).tolist(),
      "router_weight_pca_2d": router_pca2.round(4).tolist(),
      "weight_pca_spread": _pca_spread(weight_pca2),
      "router_weight_pca_spread": _pca_spread(router_pca2),
  }


def _layer_spec_to_dict(
    ent_metrics: Dict[str, torch.Tensor],
    output_sim: float,
    eff_rank: float,
    cka_pca: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
  out: Dict[str, Any] = {
      "mean_expert_entropy": float(ent_metrics["mean_expert_entropy"].item()),
      "mean_effective_num_classes": float(ent_metrics["mean_effective_num_classes"].item()),
      "mean_routing_purity": float(ent_metrics["mean_routing_purity"].item()),
      "expert_entropy": [float(x) for x in ent_metrics["expert_entropy"].cpu().tolist()],
      "expert_effective_num_classes": [
          float(x) for x in ent_metrics["expert_effective_num_classes"].cpu().tolist()
      ],
      "expert_routing_purity": [
          float(x) for x in ent_metrics["expert_routing_purity"].cpu().tolist()
      ],
      "expert_output_similarity": output_sim,
      "expert_output_effective_rank": eff_rank,
  }
  if cka_pca:
    out.update(cka_pca)
  return out


@torch.no_grad()
def _eval_accuracy_on_loader(
    model: CifarGDDSMoEBackbone,
    loader: DataLoader,
    dev: torch.device,
    max_batches: Optional[int] = None,
    mask_expert: Optional[Tuple[int, int]] = None,
) -> float:
  acc_sum, n = 0.0, 0
  for bi, (x, y) in enumerate(loader):
    if max_batches is not None and bi >= max_batches:
      break
    x, y = x.to(dev, non_blocking=False), y.to(dev, non_blocking=False)
    logits, _, _ = model(x, mask_expert=mask_expert)
    bs = int(y.numel())
    n += bs
    acc_sum += accuracy(logits, y) * bs
  return acc_sum / max(n, 1)


@torch.no_grad()
def _functional_spec_per_layer(
    model: CifarGDDSMoEBackbone,
    x: torch.Tensor,
) -> List[Tuple[float, float, Dict[str, Any]]]:
  """output similarity، effective rank، CKA/PCA برای ورودی هر لایه MoE."""
  out: List[Tuple[float, float, Dict[str, Any]]] = []
  if model.cfg.use_hybrid_spatial_moe:
    tokens = model.encode_spatial(x)
    if model.spatial_pos_embed is not None:
      tokens = tokens + model.spatial_pos_embed
    b, t, d = tokens.shape
    h = tokens.reshape(b * t, d)
    n_sp = len(model.spatial_moe_layers)
  else:
    h = model.encode(x)
    n_sp = 0

  for li, layer in enumerate(model.moe_layers):
    if model.cfg.use_hybrid_spatial_moe and li == n_sp:
      h = h.reshape(b, t, d).mean(dim=1)
    out.append(
        (
            _expert_output_similarity(layer, h),
            _expert_output_effective_rank(layer, h),
            _expert_cka_pca_metrics(layer, h),
        )
    )
    h, _, _, _, _, _, _ = layer(h)
  return out


@torch.no_grad()
def compute_specialization_metrics(
    model,
    cfg: BenchConfig,
    routing_counts: List[torch.Tensor],
    class_totals: torch.Tensor,
    first_batch_x: Optional[torch.Tensor],
) -> Dict[str, Any]:
  """۵ متریک specialization برای هر لایه MoE."""
  if not cfg.spec_metric_enabled:
    return {}

  n_layers = len(model.moe_layers)
  functional: List[Tuple[float, float, Dict[str, Any]]] = []
  if first_batch_x is not None:
    functional = _functional_spec_per_layer(model, first_batch_x)

  layer_out: Dict[str, Any] = {}
  for li in range(n_layers):
    affinity = _routing_affinity_matrix(routing_counts[li], class_totals)
    ent_m = _expert_entropy_metrics(affinity)
    if li < len(functional):
      out_sim, eff_rank, cka_pca = functional[li]
    else:
      out_sim, eff_rank, cka_pca = 0.0, 0.0, {}

    layer_out[model.layer_prefix(li)] = _layer_spec_to_dict(
        ent_m, out_sim, eff_rank, cka_pca=cka_pca
    )

  summary = {
      "mean_expert_entropy": float(
          np.mean([layer_out[k]["mean_expert_entropy"] for k in layer_out])
      ),
      "mean_effective_num_classes": float(
          np.mean([layer_out[k]["mean_effective_num_classes"] for k in layer_out])
      ),
      "mean_routing_purity": float(
          np.mean([layer_out[k]["mean_routing_purity"] for k in layer_out])
      ),
      "mean_expert_output_similarity": float(
          np.mean([layer_out[k]["expert_output_similarity"] for k in layer_out])
      ),
      "mean_expert_output_effective_rank": float(
          np.mean([layer_out[k]["expert_output_effective_rank"] for k in layer_out])
      ),
      "mean_expert_output_cka_offdiag": float(
          np.mean([layer_out[k].get("expert_output_cka_offdiag", 0.0) for k in layer_out])
      ),
      "mean_weight_pca_spread": float(
          np.mean([layer_out[k].get("weight_pca_spread", 0.0) for k in layer_out])
      ),
      "mean_router_weight_pca_spread": float(
          np.mean([layer_out[k].get("router_weight_pca_spread", 0.0) for k in layer_out])
      ),
  }
  return {"layers": layer_out, "summary": summary}


# =============================================================================
# 8) آموزش
# =============================================================================


@torch.no_grad()
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
  return float((logits.argmax(1) == y).float().mean().item())


@torch.no_grad()
def accuracy_top5(logits: torch.Tensor, y: torch.Tensor) -> float:
  max_k = min(5, logits.size(-1))
  _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
  ok = (pred == y.view(-1, 1)).any(dim=1)
  return float(ok.float().mean().item())

@torch.no_grad()
def accuracy_top2(logits: torch.Tensor, y: torch.Tensor) -> float:
  max_k = min(2, logits.size(-1))
  _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
  ok = (pred == y.view(-1, 1)).any(dim=1)
  return float(ok.float().mean().item())


def train_one_epoch(
    model: CifarGDDSMoEBackbone,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    dev: torch.device,
    cfg: BenchConfig,
) -> Tuple[float, float, float, float, Dict[str, float], Dict[str, float]]:
  model.train()
  n_moe = len(model.moe_layers)
  m_train_layers: List[MoEAuxMetrics] = [MoEAuxMetrics() for _ in range(n_moe)]
  ce_sum = 0.0
  aux_epoch: Dict[str, float] = {}
  div_epoch: Dict[str, float] = {}
  n = 0
  acc1 = 0.0
  acc2 = 0.0
  acc5 = 0.0
  need_trace = bool(
      cfg.use_grad_div or cfg.use_router_grad_div or cfg.use_router_div
  )
  for x, y in loader:
    x, y = x.to(dev, non_blocking=False), y.to(dev, non_blocking=False)
    opt.zero_grad(set_to_none=True)
    if need_trace:
      logits, aux_list, _stats, routing, trace = model(
          x, return_router=True, return_moe_trace=True
      )
    else:
      logits, aux_list, _stats, routing = model(x, return_router=True)
      trace = None
    loss = F.cross_entropy(logits, y, label_smoothing=cfg.train_label_smoothing)
    aux_total = logits.new_tensor(0.0)
    for li, ad in enumerate(aux_list):
      aux_total = aux_total + sum_weighted_aux_fixed(
          ad, model.moe_config_for_layer(li), logits
      )
    task_total = loss + aux_total
    grad_div_total = logits.new_tensor(0.0)
    router_grd_total = logits.new_tensor(0.0)
    router_div_total = logits.new_tensor(0.0)
    div_stats: Dict[str, float] = {}
    if need_trace and trace is not None:
      grad_div_total, router_grd_total, div_stats = compute_diversity_losses(
          model, trace, task_total, cfg
      )
      if cfg.use_router_div:
        router_div_total, router_div_stats = compute_router_div_losses(
            model, trace, cfg
        )
        div_stats.update(router_div_stats)
    total_loss = (
        task_total
        + cfg.grad_div_coef * grad_div_total
        + cfg.router_grad_div_coef * router_grd_total
        + cfg.router_div_coef * router_div_total
    )
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    bs = int(y.numel())
    n += bs
    ce_sum += float(loss.detach().item()) * bs
    with torch.no_grad():
      acc1 += accuracy(logits.detach(), y) * bs
      acc2 += accuracy_top2(logits.detach(), y) * bs
      acc5 += accuracy_top5(logits.detach(), y) * bs
      for li, (_rlog, scaled_r, tidx) in enumerate(routing):
        moe_li = model.moe_config_for_layer(li)
        lb_t = load_balance_loss(
            scaled_r, tidx, moe_li.num_experts, moe_li.top_k
        )
        m_train_layers[li].accumulate(scaled_r, tidx, lb_t)
    for li, ad in enumerate(aux_list):
      for k, v in ad.items():
        key = f"{model.layer_prefix(li)}_{k}"
        aux_epoch[key] = aux_epoch.get(key, 0.0) + float(v.detach().item()) * bs
    for k, v in div_stats.items():
      div_epoch[k] = div_epoch.get(k, 0.0) + float(v) * bs
  out_aux = {k: v / max(n, 1) for k, v in aux_epoch.items()}
  out_aux.update({k: v / max(n, 1) for k, v in div_epoch.items()})
  layer_dicts = [m.to_dict() for m in m_train_layers]
  mean_moe = _mean_moe_dict([d for d in layer_dicts if d])
  return (
      ce_sum / max(n, 1),
      acc1 / max(n, 1),
      acc2 / max(n, 1),
      acc5 / max(n, 1),
      out_aux,
      mean_moe,
  )


@torch.no_grad()
def eval_epoch(
    model: CifarGDDSMoEBackbone,
    loader: DataLoader,
    dev: torch.device,
    cfg: Optional[BenchConfig] = None,
) -> Tuple[float, float, float, float, Dict[str, float]]:
  model.eval()
  num_classes = model.cfg.num_classes
  n_layers = len(model.moe_layers)
  agg: Dict[str, Any] = {}
  ce_sum, n = 0.0, 0
  acc1 = 0.0
  acc2 = 0.0
  acc5 = 0.0

  m_eval_layers: List[MoEAuxMetrics] = [MoEAuxMetrics() for _ in range(n_layers)]
  routing_counts: Optional[List[torch.Tensor]] = None
  class_totals: Optional[torch.Tensor] = None
  first_batch_x: Optional[torch.Tensor] = None
  if cfg is not None and cfg.spec_metric_enabled:
    routing_counts = [
        torch.zeros(
            model.num_experts_for_layer(li), num_classes, device=dev, dtype=torch.float64
        )
        for li in range(n_layers)
    ]
    class_totals = torch.zeros(num_classes, device=dev, dtype=torch.float64)

  for x, y in loader:
    x, y = x.to(dev, non_blocking=False), y.to(dev, non_blocking=False)
    if first_batch_x is None:
      first_batch_x = x
    logits, _aux, stats_list, routing = model(x, return_router=True)
    for li, (_rlog, scaled_r, tidx) in enumerate(routing):
      moe_li = model.moe_config_for_layer(li)
      lb_t = load_balance_loss(
          scaled_r, tidx, moe_li.num_experts, moe_li.top_k
      )
      m_eval_layers[li].accumulate(scaled_r, tidx, lb_t)
      if routing_counts is not None and class_totals is not None:
        y_route = y
        if (
            model.cfg.use_hybrid_spatial_moe
            and li < len(model.spatial_moe_layers)
            and int(y.numel()) > 0
        ):
          tokens_per_image = int(tidx.shape[0]) // int(y.numel())
          if tokens_per_image > 1:
            y_route = y.repeat_interleave(tokens_per_image)
        _accumulate_routing_class_counts(
            routing_counts[li], class_totals, y_route, tidx, update_class_totals=(li == 0)
        )
    loss = F.cross_entropy(logits, y)
    bs = int(y.numel())
    n += bs
    ce_sum += float(loss.item()) * bs
    acc1 += accuracy(logits, y) * bs
    acc2 += accuracy_top2(logits, y) * bs
    acc5 += accuracy_top5(logits, y) * bs
    for li, st in enumerate(stats_list):
      _accumulate_layer_dispatch(
          agg, st, model.num_experts_for_layer(li), layer_idx=li
      )
  layer_specs = [
      (li, model.layer_prefix(li), model.num_experts_for_layer(li))
      for li in range(n_layers)
  ]
  dispatch = _finalize_dispatch_metrics(agg, layer_specs)
  eval_moe_mean = _mean_moe_dict([d for d in (m.to_dict() for m in m_eval_layers) if d])
  dispatch["eval_moe_mean_over_layers"] = eval_moe_mean

  if cfg is not None and cfg.spec_metric_enabled and routing_counts is not None:
    assert class_totals is not None
    dispatch["specialization"] = compute_specialization_metrics(
        model,
        cfg,
        routing_counts,
        class_totals,
        first_batch_x,
    )

  if n <= 0:
    return 0.0, 0.0, 0.0, 0.0, dispatch
  scale = float(n)
  return ce_sum / scale, acc1 / scale, acc2 / scale, acc5 / scale, dispatch

def make_lr_scheduler(
    opt: torch.optim.Optimizer, cfg: BenchConfig
) -> torch.optim.lr_scheduler.LRScheduler:
  """CosineAnnealingLR یا warmup خطی + cosine."""
  warmup = max(int(cfg.lr_warmup_epochs), 0)
  eta_min = float(cfg.lr_min)
  if warmup <= 0 or warmup >= cfg.epochs:
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(cfg.epochs, 1), eta_min=eta_min
    )
  cosine_epochs = max(cfg.epochs - warmup, 1)
  warmup_sched = torch.optim.lr_scheduler.LinearLR(
      opt,
      start_factor=1.0 / warmup,
      end_factor=1.0,
      total_iters=warmup,
  )
  cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
      opt, T_max=cosine_epochs, eta_min=eta_min
  )
  return torch.optim.lr_scheduler.SequentialLR(
      opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup]
  )


# =============================================================================
# 7.6) ذخیره مدل (Checkpoint)
# =============================================================================


def _format_tokens_short(n: float) -> str:
  x = float(n)
  if abs(x) >= 1_000_000:
    return f"{x / 1_000_000:.1f}M"
  if abs(x) >= 1000:
    return f"{x / 1000:.0f}k"
  return f"{x:.0f}"


def _append_epoch_metric_suffix(
    line: str,
    *,
    te_st: Dict[str, Any],
    train_moe_mean: Dict[str, float],
    spec_sum: Dict[str, Any],
) -> str:
  eval_moe = te_st.get("eval_moe_mean_over_layers") or {}
  line += (
      f"  router_entropy_tr={float(train_moe_mean.get('routing_mean_entropy', 0.0)):.3f}"
      f" router_entropy_val={float(eval_moe.get('routing_mean_entropy', 0.0)):.3f}"
      f" tokμ={_format_tokens_short(float(te_st.get('mean_tokens_per_expert', 0.0)))}"
      f" tokσ={_format_tokens_short(float(te_st.get('std_tokens_per_expert', 0.0)))}"
  )
  if spec_sum:
    line += (
        f"  expert_entropy={float(spec_sum.get('mean_expert_entropy', 0.0)):.3f}"
        f" router_purity={float(spec_sum.get('mean_routing_purity', 0.0)):.3f}"
        f" output_sim={float(spec_sum.get('mean_expert_output_similarity', 0.0)):.3f}"
        f" effective_rank={float(spec_sum.get('mean_expert_output_effective_rank', 0.0)):.3f}"
        f" cka={float(spec_sum.get('mean_expert_output_cka_offdiag', 0.0)):.3f}"
        f" pca={float(spec_sum.get('mean_weight_pca_spread', 0.0)):.3f}"
    )
  return line


def _checkpoint_path(cfg: BenchConfig) -> Optional[str]:
  """مسیر فایل checkpoint بهترین مدل بر اساس تنظیمات."""
  if cfg.checkpoint_dir is None:
    return None
  dirname = cfg.checkpoint_dir
  if cfg.checkpoint_filename:
    fname = cfg.checkpoint_filename
  elif cfg.log_json:
    base = os.path.splitext(os.path.basename(cfg.log_json))[0]
    fname = f"{base}_seed{cfg.seed}_best.pt"
  else:
    fname = f"gdds_moe_seed{cfg.seed}_best.pt"
  return os.path.join(dirname, fname)


def save_checkpoint(
    model: nn.Module,
    cfg: BenchConfig,
    epoch: int,
    best_acc: float,
) -> Optional[str]:
  """ذخیرهٔ بهترین وزن‌ها (state_dict + bench_config) برای inference / تحلیل."""
  path = _checkpoint_path(cfg)
  if path is None:
    return None
  parent = os.path.dirname(path)
  if parent:
    os.makedirs(parent, exist_ok=True)
  state = {
      "epoch": int(epoch),
      "best_test_top1": float(best_acc),
      "seed": int(cfg.seed),
      "state_dict": {k: v.cpu().clone() for k, v in model.state_dict().items()},
      "bench_config": asdict(cfg),
  }
  torch.save(state, path)
  return path


def load_checkpoint(
    model: nn.Module,
    path: str,
    dev: torch.device,
) -> Dict[str, Any]:
  """بارگذاری وزن‌های checkpoint (بدون optimizer/scheduler)."""
  state = torch.load(path, map_location=dev, weights_only=False)
  sd = state.get("state_dict") or state.get("best_model_state_dict") or state.get("model_state_dict")
  if sd is None:
    raise KeyError("checkpoint باید state_dict / best_model_state_dict / model_state_dict داشته باشد")
  model.load_state_dict(sd)
  return state


def _run_benchmark_inner(cfg: BenchConfig) -> None:
  set_reproducibility(cfg.seed)
  dev = torch.device(
      "cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu"
  )
  print_full_config(cfg)
  train_ld, test_ld = loaders(cfg)
  set_reproducibility(cfg.seed)
  model = CifarGDDSMoEBackbone(cfg).to(dev)
  opt = torch.optim.AdamW(
      adamw_param_groups(model, cfg.weight_decay),
      lr=cfg.lr,
      betas=cfg.adam_betas,
      eps=cfg.adam_eps,
  )
  sched = make_lr_scheduler(opt, cfg)
  start_ep = 0
  best_eval_top1 = float("-inf")
  best_state: Optional[Dict[str, torch.Tensor]] = None
  epochs_no_improve = 0
  patience = cfg.early_stop_patience
  min_delta = cfg.early_stop_min_delta
  best_ckpt_path: Optional[str] = None

  resume_p = cfg.resume_training_path
  resume_norm = _normalize_save_path(resume_p, "training_last.pt") if resume_p else None
  if resume_norm and os.path.isfile(resume_norm):
    start_ep, best_eval_top1, best_state, epochs_no_improve = load_training_checkpoint(
        resume_norm, model, opt, sched, dev, cfg
    )
    print(
        f"چک‌پوینت آموزش بارگذاری شد: {resume_norm}\n"
        f"  → ادامه از اپوک {start_ep + 1}/{cfg.epochs} "
        f"({start_ep} اپوک قبلاً تمام شده)  |  best_eval_top1={best_eval_top1:.4f}",
        flush=True,
    )
    if start_ep >= cfg.epochs:
      print(
          f"طبق چک‌پوینت، همهٔ {cfg.epochs} اپوک انجام شده است. "
          f"برای ادامه، مقدار `epochs` را بزرگ‌تر از {cfg.epochs} بگذار.",
          flush=True,
      )
      return

  log_path = _normalize_log_json_path(cfg.log_json)
  log_nd = max(int(cfg.log_json_float_decimals), 0)
  best_log_row: Optional[Dict[str, Any]] = None
  best_log_epoch: Optional[int] = None

  train_cp_norm = _normalize_save_path(cfg.training_checkpoint_path, "training_last.pt")

  for ep in range(start_ep, cfg.epochs):
    tr_ce, tr_a1,tr_a2, tr_a5, tr_aux, train_moe_mean = train_one_epoch(model, train_ld, opt, dev, cfg)
    te_ce, te_a1,te_a2, te_a5, te_st = eval_epoch(model, test_ld, dev, cfg)
    sched.step()
    improved = te_a1 > best_eval_top1 + min_delta
    if improved:
      best_eval_top1 = te_a1
      epochs_no_improve = 0
      best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
      ckpt = save_checkpoint(model, cfg, ep + 1, best_eval_top1)
      if ckpt:
        best_ckpt_path = ckpt
    else:
      epochs_no_improve += 1
    es_left = (patience - epochs_no_improve) if (patience is not None and patience > 0) else None
    row = {
        "epoch": ep + 1,
        "train_ce": tr_ce,
        "train_top1": tr_a1,
        "train_top2": tr_a2,
        "train_top5": tr_a5,
        "test_ce": te_ce,
        "test_top1": te_a1,
        "test_top2": te_a2,
        "test_top5": te_a5,
        "lr": float(opt.param_groups[0]["lr"]),
        "train_aux_mean": _train_aux_for_log(tr_aux),
        "train_moe_mean_over_layers": train_moe_mean,
        "eval_moe_mean_over_layers": te_st.get("eval_moe_mean_over_layers") or {},
        "eval_dispatch": te_st,
        "best_eval_top1_so_far": best_eval_top1,
        "early_stop_patience_left": (es_left if es_left is not None else -1),
    }
    if improved and log_path:
      best_log_row = row
      best_log_epoch = ep + 1
      _write_best_epoch_log(log_path, row, asdict(cfg), log_nd)
    
    line = (
        f"ep {ep+1}/{cfg.epochs}  tr@1={tr_a1:.4f}  tr@2={tr_a2:.4f}  tr@5={tr_a5:.4f}  "
        f"te@1={te_a1:.4f} te@2={te_a2:.4f} te@5={te_a5:.4f}  "
        f"dr~{float(te_st.get('drop_rate', 0.0)):.3f} util~{float(te_st.get('capacity_usage_ratio', 0.0)):.3f}"
    )
    if es_left is not None:
      line += f"  es={es_left}"
    spec_sum = (te_st.get("specialization") or {}).get("summary") or {}
    line = _append_epoch_metric_suffix(
        line,
        te_st=te_st,
        train_moe_mean=train_moe_mean,
        spec_sum=spec_sum,
    )
    if cfg.use_router_div:
      rdiv = float(tr_aux.get("router_div_total", 0.0))
      line += f"  rdiv={rdiv:.4f}"
    print(line, flush=True)

    if train_cp_norm:
      save_training_checkpoint(
          train_cp_norm,
          cfg,
          model,
          opt,
          sched,
          epochs_completed=ep + 1,
          best_eval_top1=best_eval_top1,
          best_model_state=best_state,
          epochs_no_improve=epochs_no_improve,
      )

    if patience is not None and patience > 0 and epochs_no_improve >= patience:
      print(
          f"early stop: eval top1 برای {patience} اپوک بهبود نداشت "
          f"(best={best_eval_top1:.4f}, min_delta={min_delta})",
          flush=True,
      )
      break

  if log_path and best_log_row is not None:
    print(
        f"[log] آخرین جزئیات اپوک best روی دیسک: ep={best_log_epoch}, "
        f"te@1={best_log_row.get('test_top1'):.4f} → {log_path}",
        flush=True,
    )

  best_path = _normalize_save_path(cfg.checkpoint_path, "best_model.pt")
  if best_path and best_state is not None:
    bp = os.path.dirname(os.path.abspath(best_path))
    if bp:
      os.makedirs(bp, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "best_eval_top1": best_eval_top1,
        "bench_config": asdict(cfg),
    }
    torch.save(payload, best_path)
    print(f"بهترین وزنها ذخیره شد (eval top1={best_eval_top1:.4f}): {best_path}", flush=True)

  if best_ckpt_path:
    print(
        f"[checkpoint] بهترین مدل ذخیره شد: {best_ckpt_path} "
        f"(test_top1={best_eval_top1:.4f})",
        flush=True,
    )


def run_benchmark(cfg: Union[BenchConfig, MoEConfig]) -> None:
  cfg = as_bench_config(cfg)
  console_path = _default_console_path(cfg)
  with console_logging(console_path):
    if console_path:
      print(f"[console-log] خروجی کنسول در فایل ذخیره می‌شود: {console_path}", flush=True)
    _run_benchmark_inner(cfg)


# =============================================================================
# 9) Presetها
#
# - تابع‌های preset_moe_* فقط MoEConfig برمی‌گردانند (برای ترکیب دستی).
# - تابع‌های preset_bench_* همیشه BenchConfig کامل هستند → مستقیم run_benchmark.
# - تقویت داده / label smoothing پیش‌فرض BenchConfig برای همهٔ presetها یکسان است.
# =============================================================================

def preset_bench_gdds_tiny_imagenet200_hybrid_b() -> BenchConfig:
  """Run B: GDDS — Hybrid Spatial (1×) → GAP → Global (1×); backbone stage3: 384ch × 3 blocks."""
  global_moe = MoEConfig(
      num_experts=16,
      top_k=2,
      expert_hidden=512,
      expert_dropout=0.2,
      router_noise_std=0.02,
      capacity_factor=2,
      use_load_balance=True,
      load_balance_coef=0.00001,
      use_z_loss=True,
      z_loss_coef=1e-4,
  )
  spatial_moe = MoEConfig(
      num_experts=16,
      top_k=2,
      expert_hidden=512,
      expert_dropout=0.2,
      router_noise_std=0.02,
      capacity_factor=2,
      use_load_balance=True,
      load_balance_coef=0.00001,
      use_z_loss=True,
      z_loss_coef=1e-4,
  )
  return BenchConfig(
      moe=global_moe,
      spatial_moe=spatial_moe,
      use_hybrid_spatial_moe=True,
      moe_num_spatial_layers=1,
      moe_num_global_layers=1,
      use_spatial_pos_embed=True,
      dim=384,
      feature_channels=384,
      backbone_stage3_depth=3,
      use_grad_div=True,
      grad_div_coef=0.01,
      grad_div_global_only=False,
      grad_div_spatial_only=True,
      grad_div_max_tokens=128,
      use_router_div=False,
      use_router_grad_div=False,
      num_classes=200,
      image_size=64,
      epochs=400,
      lr=1.2e-3,
      weight_decay=0.05,
      adam_eps=1e-8,
      lr_warmup_epochs=5,
      batch_size=64,
      seed=40,
      early_stop_patience=12,
      early_stop_min_delta=0.0001,
      spec_ablation_max_batches=50,
      training_checkpoint_path=None,
      checkpoint_dir="checkpoints",
      resume_training_path=None,
      log_json="gdd_tiny_img200_2l_0.01_Sb_0.0001_40.json",
  )


def preset_bench_gdds_tiny_imagenet200_hybrid() -> BenchConfig:
  """Alias: preset پیش‌فرض = Run B (GDDS only)."""
  return preset_bench_gdds_tiny_imagenet200_hybrid_b()




if __name__ == "__main__":
  print("start: ", time.time())
  bench = preset_bench_gdds_tiny_imagenet200_hybrid()
  bench.checkpoint_filename = "gdd_tiny_img200_Sl_0.01__lb_0.0001_40.pt"
  run_benchmark(bench)
  print("end: ", time.time())

