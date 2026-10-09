# -*- coding: utf-8 -*-
"""MoE hybrid روی MNIST — همان معماری moe_cif10 با stem تک‌کاناله و لودر MNIST.

برای expert_analysis / visualize؛ آموزش اصلی در نوت‌بوک‌های mnist/A1_* است.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

from moe_cif10 import (  # noqa: F401
    BenchConfig as _CifarBenchConfig,
    MoEConfig,
    CifarSparseMoEBackbone as _CifarSparseMoEBackbone,
    _dataloader_seed_worker,
    _normalize_save_path,
    _seed_torchvision_random,
    set_reproducibility,
)


@dataclass
class BenchConfig(_CifarBenchConfig):
    in_channels: int = 1
    image_size: int = 32


class CifarSparseMoEBackbone(_CifarSparseMoEBackbone):
    """همان backbone CIFAR؛ stem و dummy spatial از in_channels / image_size می‌آید."""

    def __init__(self, cfg: BenchConfig) -> None:
        super().__init__(cfg)
        in_ch = int(getattr(cfg, "in_channels", 1))
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self._init_conv_weights()
        if cfg.use_hybrid_spatial_moe:
            self._infer_spatial_tokens()

    def _infer_spatial_tokens(self) -> int:
        # Parent __init__ still has the CIFAR 3-ch stem; after we replace it,
        # this is called again with the real 1-ch stem.
        ch = int(self.stem[0].in_channels)
        sz = int(getattr(self.cfg, "image_size", 32))
        with torch.no_grad():
            feat = self.encode_features(torch.zeros(1, ch, sz, sz))
        self._spatial_hw = (int(feat.shape[2]), int(feat.shape[3]))
        n_tokens = int(feat.shape[2] * feat.shape[3])
        self._spatial_tokens = n_tokens
        return n_tokens


def loaders(cfg: BenchConfig) -> Tuple[DataLoader, DataLoader]:
    set_reproducibility(cfg.seed)
    _seed_torchvision_random(cfg.seed)
    mean = (0.1307,)
    std = (0.3081,)
    sz = int(getattr(cfg, "image_size", 32))
    resize = T.Resize(sz)
    train_tfm_list = [resize]
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
    test_tfm = T.Compose([resize, T.ToTensor(), T.Normalize(mean, std)])
    g_train = torch.Generator().manual_seed(cfg.seed)
    g_test = torch.Generator().manual_seed(cfg.seed + 10_000)
    w_init = _dataloader_seed_worker if cfg.num_workers > 0 else None
    tr = torchvision.datasets.MNIST(
        cfg.data_dir, True, download=True, transform=train_tfm
    )
    te = torchvision.datasets.MNIST(
        cfg.data_dir, False, download=True, transform=test_tfm
    )
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
