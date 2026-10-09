# Copyright 2025
# -*- coding: utf-8 -*-
"""
پلتفرم یک‌فایلهٔ Sparse MoE (compute-sparse) برای CIFAR-100 — مناسب Colab.

هدف: **base تمیز + همهٔ لاس‌های کمکی و محدودیت‌ها قابل خاموش** برای ablation / benchmark.

بخش‌ها (به ترتیب):
  1) MoEConfig / BenchConfig   2) Expert   3) لاس‌های کمکی ماژولار
  4) dispatch مرتب‌شده (≈ O(B·K) گروه‌بندی per-expert) + capacity اختیاری
  5) SparseMoELayer   6) Backbone   7) دادهٔ CIFAR-100   8) آموزش ساده
  9) presetهای آماده

اجرا:  python new_files/sparse_moe_experiment_colab.py
یا در Colab:  %run path/to/sparse_moe_experiment_colab.py
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
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


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
  num_classes: int = 100
  dim: int = 256
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
  local_cifar100: Optional[str] = 'cifar-100-python.tar.gz'
  train_random_crop_pad: int = 4
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

  # --- Hybrid: Spatial MoE (قبل از GAP) → GAP → Global MoE ---
  use_hybrid_spatial_moe: bool = False
  spatial_moe: Optional[MoEConfig] = None
  moe_num_spatial_layers: int = 1
  moe_num_global_layers: int = 2
  use_spatial_pos_embed: bool = True
  feature_channels: int = 256


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
    if lc.get("dim") != cfg.dim or lc.get("num_classes") != cfg.num_classes:
      print(
          "هشدار: dim یا num_classes با چک‌پوینت فرق دارد؛ بارگذاری ممکن است نامعتبر باشد.",
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
  """یک فایل JSON با جزئیات اپوک best (بازنویسی کامل)."""
  os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
  nd = max(int(float_decimals), 0)
  payload = {
      "log_format": "best_epoch_v1",
      "bench_config": bench_config,
      **epoch_row,
  }
  with open(log_path, "w", encoding="utf-8") as f:
    json.dump(round_json_floats(payload, nd), f, indent=2, ensure_ascii=False)

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
  ) -> Tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      Dict[str, torch.Tensor],
      Dict[str, Any],
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
    return y, router_logits, scaled, topk_idx, aux, d_stats


# =============================================================================
# 6) Backbone CIFAR-100
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


class CifarSparseMoEBackbone(nn.Module):
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

    self.stage3 = nn.Sequential(
        BasicBlock(128, 256, stride=2),
        BasicBlock(256, 256),
    )

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
      feat = self.encode_features(torch.zeros(1, 3, 32, 32))
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
      mask_expert: Optional[Tuple[int, int]] = None,
  ):
    aux_list: List[Dict[str, torch.Tensor]] = []
    stats_list: List[Dict[str, Any]] = []
    routing: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

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
      h, rlog, scaled, tidx, aux, st = layer(h, mask_expert_id=mask_id)
      aux_list.append(aux)
      stats_list.append(st)
      if return_router:
        routing.append((rlog, scaled, tidx))

    h = self.final_norm(h)
    logits = self.head(h)

    if return_router:
      return logits, aux_list, stats_list, routing

    return logits, aux_list, stats_list


# =============================================================================
# 7) دادهٔ CIFAR-100 (لوکال سبک)
# =============================================================================


def _cifar_ok(p: str) -> bool:
  return all(os.path.isfile(os.path.join(p, n)) for n in ("train", "test", "meta"))


def ensure_cifar100(root: str, local: Optional[str] = None) -> None:
  tgt = os.path.join(root, "cifar-100-python")
  if _cifar_ok(tgt):
    return
  os.makedirs(root, exist_ok=True)
  u = (
      os.path.abspath(os.path.expanduser(str(local).strip().strip('"')))
      if local and str(local).strip()
      else None
  )
  if u and os.path.isdir(u) and _cifar_ok(u):
    dst = tgt
    if not os.path.exists(dst):
      shutil.copytree(u, dst)
    return
  if u and os.path.isfile(u):
    try:
      with tarfile.open(u, "r:*") as tf:
        tf.extractall(path=root)
    except (OSError, tarfile.TarError):
      pass
  for name in ("cifar-100-python.tar", "cifar-100-python.tar.gz"):
    p = os.path.join(root, name)
    if os.path.isfile(p):
      try:
        with tarfile.open(p, "r:*") as tf:
          tf.extractall(path=root)
      except (OSError, tarfile.TarError):
        pass
    if _cifar_ok(tgt):
      return


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
  ensure_cifar100(cfg.data_dir, cfg.local_cifar100)
  mean = (0.5071, 0.4867, 0.4408)
  std = (0.2675, 0.2565, 0.2761)
  train_tfm_list: List[nn.Module] = []
  if cfg.train_random_crop_pad > 0:
    train_tfm_list.append(T.RandomCrop(32, padding=cfg.train_random_crop_pad))
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
  tr = torchvision.datasets.CIFAR100(
      cfg.data_dir, True, download=True, transform=train_tfm
  )
  te = torchvision.datasets.CIFAR100(
      cfg.data_dir, False, download=True, transform=test_tfm
  )
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
  cov = (y.T @ y) / float(n)
  eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)
  s1 = eigvals.sum()
  s2 = eigvals.pow(2).sum()
  if float(s2.item()) < 1e-12:
    return 0.0
  return float((s1 * s1 / s2).item())


def _layer_spec_to_dict(
    ent_metrics: Dict[str, torch.Tensor],
    output_sim: float,
    eff_rank: float,
) -> Dict[str, Any]:
  return {
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


@torch.no_grad()
def _eval_accuracy_on_loader(
    model: CifarSparseMoEBackbone,
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
    model: CifarSparseMoEBackbone,
    x: torch.Tensor,
) -> List[Tuple[float, float]]:
  """output similarity و effective rank برای ورودی هر لایه MoE."""
  out: List[Tuple[float, float]] = []
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
        )
    )
    h, _, _, _, _, _ = layer(h)
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
  functional: List[Tuple[float, float]] = []
  if first_batch_x is not None:
    functional = _functional_spec_per_layer(model, first_batch_x)

  layer_out: Dict[str, Any] = {}
  for li in range(n_layers):
    affinity = _routing_affinity_matrix(routing_counts[li], class_totals)
    ent_m = _expert_entropy_metrics(affinity)
    out_sim, eff_rank = functional[li] if li < len(functional) else (0.0, 0.0)

    layer_out[model.layer_prefix(li)] = _layer_spec_to_dict(
        ent_m, out_sim, eff_rank
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
    model: CifarSparseMoEBackbone,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    dev: torch.device,
    cfg: BenchConfig,
) -> Tuple[float, float, float, Dict[str, float], Dict[str, float]]:
  model.train()
  n_moe = len(model.moe_layers)
  m_train_layers: List[MoEAuxMetrics] = [MoEAuxMetrics() for _ in range(n_moe)]
  ce_sum = 0.0
  aux_epoch: Dict[str, float] = {}
  n = 0
  acc1 = 0.0
  acc2 = 0.0
  acc5 = 0.0
  for x, y in loader:
    x, y = x.to(dev, non_blocking=False), y.to(dev, non_blocking=False)
    opt.zero_grad(set_to_none=True)
    logits, aux_list, _stats, routing = model(x, return_router=True)
    loss = F.cross_entropy(logits, y, label_smoothing=cfg.train_label_smoothing)
    aux_total = logits.new_tensor(0.0)
    for li, ad in enumerate(aux_list):
      aux_total = aux_total + sum_weighted_aux_fixed(
          ad, model.moe_config_for_layer(li), logits
      )
    (loss + aux_total).backward()
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
  out_aux = {k: v / max(n, 1) for k, v in aux_epoch.items()}
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
    model: CifarSparseMoEBackbone,
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
    if routing_counts is not None and class_totals is not None:
      logits, _aux, stats_list, routing = model(x, return_router=True)
      for li, (_rlog, _scaled, tidx) in enumerate(routing):
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
    else:
      logits, _aux, stats_list = model(x)
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
    fname = f"backbone_moe_seed{cfg.seed}_best.pt"
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
  model = CifarSparseMoEBackbone(cfg).to(dev)
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
        "eval_dispatch": te_st,
        "best_eval_top1_so_far": best_eval_top1,
        "early_stop_patience_left": (es_left if es_left is not None else -1),
    }
    if improved:
      best_log_row = row
      best_log_epoch = ep + 1
    line = (
        f"ep {ep+1}/{cfg.epochs}  tr@1={tr_a1:.4f}  tr@2={tr_a2:.4f}  tr@5={tr_a5:.4f}  "
        f"te@1={te_a1:.4f} te@2={te_a2:.4f} te@5={te_a5:.4f}  "
        f"dr~{float(te_st.get('drop_rate', 0.0)):.3f} util~{float(te_st.get('capacity_usage_ratio', 0.0)):.3f}"
    )
    if es_left is not None:
      line += f"  es={es_left}"
    spec_sum = (te_st.get("specialization") or {}).get("summary") or {}
    if spec_sum:
      line += (
          f"  spec_H={float(spec_sum.get('mean_expert_entropy', 0.0)):.3f}"
          f" pur={float(spec_sum.get('mean_routing_purity', 0.0)):.3f}"
          f" sim={float(spec_sum.get('mean_expert_output_similarity', 0.0)):.3f}"
      )
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
    nd = max(int(cfg.log_json_float_decimals), 0)
    _write_best_epoch_log(log_path, best_log_row, asdict(cfg), nd)
    print(
        f"[log] جزئیات اپوک best (ep={best_log_epoch}, te@1={best_log_row.get('test_top1'):.4f}) "
        f"→ {log_path}",
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

def preset_bench_cifar100_repro() -> BenchConfig:
  """همان preset سریع، بدون augmentation / dropout / router noise — برای بنچمارک تکرارپذیر."""
  cfg = preset_bench_cifar100_fast()
  cfg.seed = 41
  cfg.moe.router_noise_std = 0.0
  cfg.moe.expert_dropout = 0.0
  cfg.train_random_crop_pad = 0
  cfg.train_hflip = False
  cfg.train_randaugment_ops = 0
  cfg.resume_training_path = None
  return cfg


def preset_bench_cifar100_hybrid() -> BenchConfig:
  """Spatial MoE (1×) → GAP → Global MoE (2×) — معماری hybrid."""
  global_moe = MoEConfig(
      num_experts=8,
      top_k=2,
      expert_hidden=384,
      expert_dropout=0.2,
      router_noise_std=0.02,
      capacity_factor=2,
      use_load_balance=True,
      load_balance_coef=0.00001,
      use_z_loss=True,
      z_loss_coef=1e-4,
  )
  spatial_moe = MoEConfig(
      num_experts=8,
      top_k=2,
      expert_hidden=384,
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
      epochs=400,
      lr=1.2e-3,
      weight_decay=0.05,
      adam_eps=1e-8,
      lr_warmup_epochs=5,
      batch_size=128,
      seed=40,
      early_stop_patience=12,
      early_stop_min_delta=0.0001,
      training_checkpoint_path=None,
      checkpoint_dir="checkpoints",
      checkpoint_filename="moe_cifar100_lbl_0.00001_40.pt",
      resume_training_path=None,
  )

if __name__ == "__main__":
  print("start: ", time.time())
  bench = preset_bench_cifar100_hybrid()
  bench.log_json = "moe_cifar100_lbl_0.00001_log_40.json"
  run_benchmark(bench)
  print("end: ", time.time())

