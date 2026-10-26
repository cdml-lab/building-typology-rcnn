"""Faster R-CNN with a ResNet-101 FPN backbone (torchvision), 12 outputs
(background + 11 network classes)."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .classes import NUM_NETWORK_CLASSES

CHECKPOINT = {
    "file": "fasterrcnnv100-12.pth",
    "version": "v100-12 (February 2026)",
    "backbone": "resnet101",
    "sha256": "87727651374c5ae9e2b652f4c01487d1565b9971bff670044b86733604c1dc78",
    "num_classes": NUM_NETWORK_CLASSES,
}


def build_model(backbone: str = "resnet101", pretrained_backbone: bool = False):
    import torch.nn  # noqa: F401
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
    weights = None
    if pretrained_backbone:                       # ImageNet initialisation for training
        from torchvision.models import ResNet101_Weights, ResNet50_Weights
        weights = (ResNet101_Weights.IMAGENET1K_V1 if backbone == "resnet101"
                   else ResNet50_Weights.IMAGENET1K_V1)
    bb = resnet_fpn_backbone(backbone_name=backbone, weights=weights)
    return FasterRCNN(bb, num_classes=NUM_NETWORK_CLASSES)


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path, device=None, verify: bool = True, backbone: str = "resnet101"):
    """Model in eval mode with the trained weights. verify=True checks the
    sha256 of the released checkpoint (pass verify=False for your own)."""
    import torch
    path = Path(path)
    if verify and path.name == CHECKPOINT["file"]:
        got = sha256(path)
        if got != CHECKPOINT["sha256"]:
            raise ValueError(f"checkpoint checksum mismatch: {got}")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = build_model(backbone)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return m.to(dev).eval(), dev
