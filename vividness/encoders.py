"""Frozen vision encoders used with the probe."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

# Name, backend, public checkpoint. Listed in the order reported for the benchmark.
MODELS: dict[str, tuple[str, str]] = {
    "siglip2_b16": ("siglip", "google/siglip2-base-patch16-224"),
    "siglip_b16": ("siglip", "google/siglip-base-patch16-224"),
    "clip_b16": ("clip", "openai/clip-vit-base-patch16"),
    "mae_b16": ("timm", "vit_base_patch16_224.mae"),
    "dinov3_b": ("hf", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    "resnet50": ("timm", "resnet50"),
    "dinov3_l": ("hf", "facebook/dinov3-vitl16-pretrain-lvd1689m"),
    "resnet18": ("timm", "resnet18"),
    "vgg16": ("timm", "vgg16"),
    "vit_b16": ("timm", "vit_base_patch16_224"),
    "dinov2_b14": ("timm", "vit_base_patch14_dinov2.lvd142m"),
    "convnext_t": ("timm", "convnext_tiny"),
}


@dataclass
class Encoder:
    backbone: nn.Module
    preprocess: Callable
    embed_dim: int


def list_models() -> str:
    return "\n".join(f"{name:16} {family:8} {checkpoint}" for name, (family, checkpoint) in MODELS.items())


def build_encoder(name: str, device: str) -> Encoder:
    if name not in MODELS:
        known = ", ".join(MODELS)
        raise SystemExit(f"unknown model {name!r}. Choose one of: {known}")
    family, checkpoint = MODELS[name]
    if family == "timm":
        return _timm(checkpoint, device)
    if family == "clip":
        return _clip(checkpoint, device)
    if family == "siglip":
        return _siglip(checkpoint, device)
    return _hf_vision(checkpoint, device)


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module


def _output_dim(module: nn.Module, height: int, width: int, device: str) -> int:
    with torch.inference_mode():
        sample = module(torch.zeros(1, 3, height, width, device=device))
    return int(sample.shape[-1])


def _timm(checkpoint: str, device: str) -> Encoder:
    import timm
    from timm.data import create_transform, resolve_data_config

    backbone = _freeze(timm.create_model(checkpoint, pretrained=True, num_classes=0)).to(device)
    config = resolve_data_config(backbone.pretrained_cfg, use_test_size=True)
    transform = create_transform(**config, is_training=False)
    channels, height, width = config.get("input_size", (3, 224, 224))

    class _Flat(nn.Module):
        def __init__(self, base: nn.Module) -> None:
            super().__init__()
            self.base = base

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            features = self.base(images)
            return features.flatten(1) if features.ndim > 2 else features

    module = _Flat(backbone).to(device)
    embed_dim = _output_dim(module, height, width, device)
    return Encoder(module, transform, embed_dim)


def _processor_size(processor) -> tuple[int, int]:
    from PIL import Image

    image = Image.new("RGB", (32, 32))
    values = processor(images=image, return_tensors="pt")["pixel_values"]
    return int(values.shape[-2]), int(values.shape[-1])


def _clip(checkpoint: str, device: str) -> Encoder:
    from transformers import CLIPImageProcessor, CLIPVisionModel

    processor = CLIPImageProcessor.from_pretrained(checkpoint)

    class _Clip(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = _freeze(CLIPVisionModel.from_pretrained(checkpoint))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            hidden = self.vision(pixel_values=images, return_dict=True).last_hidden_state
            return hidden[:, 0, :]

    module = _Clip().to(device)
    height, width = _processor_size(processor)
    embed_dim = _output_dim(module, height, width, device)

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    return Encoder(module, preprocess, embed_dim)


def _siglip(checkpoint: str, device: str) -> Encoder:
    from transformers import AutoImageProcessor, SiglipVisionModel

    processor = AutoImageProcessor.from_pretrained(checkpoint)

    class _Siglip(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision = _freeze(SiglipVisionModel.from_pretrained(checkpoint))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            output = self.vision(pixel_values=images, return_dict=True)
            if getattr(output, "pooler_output", None) is not None:
                return output.pooler_output
            return output.last_hidden_state[:, 0, :]

    module = _Siglip().to(device)
    height, width = _processor_size(processor)
    embed_dim = _output_dim(module, height, width, device)

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    return Encoder(module, preprocess, embed_dim)


def _hf_vision(checkpoint: str, device: str) -> Encoder:
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(checkpoint)

    class _Vision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = _freeze(AutoModel.from_pretrained(checkpoint))

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            output = self.model(pixel_values=images, return_dict=True)
            if getattr(output, "pooler_output", None) is not None:
                return output.pooler_output
            return output.last_hidden_state[:, 0, :]

    module = _Vision().to(device)
    height, width = _processor_size(processor)
    embed_dim = _output_dim(module, height, width, device)

    def preprocess(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    return Encoder(module, preprocess, embed_dim)
