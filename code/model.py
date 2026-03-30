# """
# model.py
# ResNet-50 backbone with Squeeze-and-Excitation (SE) blocks inserted into
# every Bottleneck unit.

# Modification summary (for report):
#     - Standard ResNet-50 Bottleneck is extended with an SE module after the
#       final BN layer but before the residual addition.
#     - SE module: GlobalAvgPool → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid
#       where reduction ratio r=16 (default).
#     - This adds channel-wise attention: the network learns to recalibrate
#       feature responses by explicitly modelling inter-channel dependencies.
#     - Parameter overhead: ~2.5 M on top of ResNet-50's 23.5 M ≈ 26 M total,
#       well within the 100 M limit.
#     - Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
# """

# from typing import Optional, Type

# import torch
# import torch.nn as nn
# from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights
# from torchvision.models.resnet import Bottleneck


# # ── SE Module ──────────────────────────────────────────────────────────────────

# class SEModule(nn.Module):
#     """
#     Squeeze-and-Excitation module.

#     Args:
#         channels:  number of input channels (C)
#         reduction: reduction ratio r (default 16)
#     """

#     def __init__(self, channels: int, reduction: int = 16) -> None:
#         super().__init__()
#         mid = max(channels // reduction, 4)   # guard against tiny channel counts
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),           # (B, C, 1, 1)
#             nn.Flatten(),                      # (B, C)
#             nn.Linear(channels, mid, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(mid, channels, bias=False),
#             nn.Sigmoid(),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
#         return x * scale


# # ── SE-Bottleneck ──────────────────────────────────────────────────────────────

# class SEBottleneck(Bottleneck):
#     """
#     Bottleneck with an SE module injected before the residual addition.

#     Inherits all conv / bn / downsample logic from torchvision's Bottleneck;
#     only forward() is overridden.
#     """

#     def __init__(self, *args, reduction: int = 16, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         # planes * expansion = output channels of this block
#         out_channels = self.conv3.out_channels
#         self.se = SEModule(out_channels, reduction=reduction)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         out = self.se(out)          # <-- channel attention before residual add

#         if self.downsample is not None:
#             identity = self.downsample(x)

#         out += identity
#         out = self.relu(out)
#         return out


# # ── Model factory ──────────────────────────────────────────────────────────────

# def _replace_bottlenecks(model: nn.Module, reduction: int = 16) -> None:
#     """
#     Walk the ResNet and swap every Bottleneck with an SEBottleneck in-place.
#     Copies all existing weights so pretrained parameters are preserved.
#     """
#     for name, module in model.named_children():
#         if isinstance(module, nn.Sequential):
#             for block_name, block in module.named_children():
#                 if isinstance(block, Bottleneck) and not isinstance(block, SEBottleneck):
#                     se_block = SEBottleneck(
#                         inplanes=block.conv1.in_channels,
#                         planes=block.conv1.out_channels,
#                         stride=block.stride,
#                         downsample=block.downsample,
#                         groups=block.conv2.groups,
#                         base_width=64,          # default for standard ResNet
#                         dilation=block.conv2.dilation
#                             if hasattr(block.conv2, "dilation") else 1,
#                         norm_layer=nn.BatchNorm2d,
#                         reduction=reduction,
#                     )
#                     # copy pretrained weights (conv + bn layers)
#                     se_block.conv1.load_state_dict(block.conv1.state_dict())
#                     se_block.conv2.load_state_dict(block.conv2.state_dict())
#                     se_block.conv3.load_state_dict(block.conv3.state_dict())
#                     se_block.bn1.load_state_dict(block.bn1.state_dict())
#                     se_block.bn2.load_state_dict(block.bn2.state_dict())
#                     se_block.bn3.load_state_dict(block.bn3.state_dict())
#                     if block.downsample is not None:
#                         se_block.downsample = block.downsample
#                     # SE weights are randomly initialised (new module)
#                     module[int(block_name)] = se_block
#         else:
#             _replace_bottlenecks(module, reduction)


# def build_model(num_classes: int = 100, reduction: int = 16) -> nn.Module:
#     """
#     Build SE-ResNet-50 with ImageNet pretrained weights.

#     Steps:
#         1. Load standard ResNet-50 (ImageNet pretrained).
#         2. Replace every Bottleneck with SEBottleneck (preserving weights).
#         3. Replace the final FC layer to output `num_classes` logits.

#     Args:
#         num_classes: number of output classes (100 for this task)
#         reduction:   SE reduction ratio

#     Returns:
#         nn.Module ready for training
#     """
#     # 1. pretrained backbone
#     # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#     model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

#     # 2. inject SE into every bottleneck
#     _replace_bottlenecks(model, reduction=reduction)

#     # 3. replace classifier head
#     in_features = model.fc.in_features
#     model.fc = nn.Sequential(
#         nn.Dropout(p=0.5),
#         nn.Linear(in_features, num_classes),
#     )

#     return model


# # ── Parameter count utility ────────────────────────────────────────────────────

# def count_parameters(model: nn.Module) -> int:
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)


# # ── Sanity check ───────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     model = build_model(num_classes=100)
#     total = count_parameters(model)
#     print(f"Trainable parameters: {total / 1e6:.2f} M")
#     assert total < 100e6, "Model exceeds 100 M parameter limit!"

#     dummy = torch.randn(4, 3, 224, 224)
#     out = model(dummy)
#     print(f"Output shape: {out.shape}")   # expect (4, 100)

#     # verify SE modules are present
#     se_count = sum(1 for m in model.modules() if isinstance(m, SEModule))
#     print(f"SE modules inserted: {se_count}")   # expect 16 for ResNet-50

# ======================================================================================

"""
model.py
ResNet-101 backbone with Squeeze-and-Excitation (SE) blocks inserted into
every Bottleneck unit.

Modification summary (for report):
    - Standard ResNet-101 Bottleneck is extended with an SE module after the
      final BN layer but before the residual addition.
    - SE module: GlobalAvgPool → FC(C→C/r) → ReLU → FC(C/r→C) → Sigmoid
      where reduction ratio r=16 (default).
    - This adds channel-wise attention: the network learns to recalibrate
      feature responses by explicitly modelling inter-channel dependencies.
    - Parameter overhead: ~2.5 M on top of ResNet-101's ~42.5 M ≈ 45 M total,
      well within the 100 M limit.
    - Reference: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet101, ResNet101_Weights
from torchvision.models.resnet import Bottleneck


# ── SE Module ──────────────────────────────────────────────────────────────────

class SEModule(nn.Module):
    """
    Squeeze-and-Excitation module.

    Args:
        channels:  number of input channels (C)
        reduction: reduction ratio r (default 16)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)   # guard against tiny channel counts
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B, C, 1, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ── SE-Bottleneck ──────────────────────────────────────────────────────────────

class SEBottleneck(Bottleneck):
    """
    Bottleneck with an SE module injected before the residual addition.

    Inherits all conv / bn / downsample logic from torchvision's Bottleneck;
    only forward() is overridden.
    """

    def __init__(self, *args, reduction: int = 16, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # planes * expansion = output channels of this block
        out_channels = self.conv3.out_channels
        self.se = SEModule(out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)          # <-- channel attention before residual add

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


# ── Model factory ──────────────────────────────────────────────────────────────

def _replace_bottlenecks(model: nn.Module, reduction: int = 16) -> None:
    """
    Walk the ResNet and swap every Bottleneck with an SEBottleneck in-place.
    Copies all existing weights so pretrained parameters are preserved.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            for block_name, block in module.named_children():
                if isinstance(block, Bottleneck) and not isinstance(block, SEBottleneck):
                    se_block = SEBottleneck(
                        inplanes=block.conv1.in_channels,
                        planes=block.conv1.out_channels,
                        stride=block.stride,
                        downsample=block.downsample,
                        groups=block.conv2.groups,
                        base_width=64,          # default for standard ResNet
                        dilation=block.conv2.dilation
                            if hasattr(block.conv2, "dilation") else 1,
                        norm_layer=nn.BatchNorm2d,
                        reduction=reduction,
                    )
                    # copy pretrained weights (conv + bn layers)
                    se_block.conv1.load_state_dict(block.conv1.state_dict())
                    se_block.conv2.load_state_dict(block.conv2.state_dict())
                    se_block.conv3.load_state_dict(block.conv3.state_dict())
                    se_block.bn1.load_state_dict(block.bn1.state_dict())
                    se_block.bn2.load_state_dict(block.bn2.state_dict())
                    se_block.bn3.load_state_dict(block.bn3.state_dict())
                    if block.downsample is not None:
                        se_block.downsample = block.downsample
                    # SE weights are randomly initialised (new module)
                    module[int(block_name)] = se_block
        else:
            _replace_bottlenecks(module, reduction)


def build_model(num_classes: int = 100, reduction: int = 16) -> nn.Module:
    """
    Build SE-ResNet-101 with ImageNet pretrained weights.

    Steps:
        1. Load standard ResNet-101 (ImageNet pretrained, V2 weights).
        2. Replace every Bottleneck with SEBottleneck (preserving weights).
        3. Replace the final FC layer to output `num_classes` logits.

    Args:
        num_classes: number of output classes (100 for this task)
        reduction:   SE reduction ratio

    Returns:
        nn.Module ready for training
    """
    # 1. pretrained backbone
    model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)

    # 2. inject SE into every bottleneck
    _replace_bottlenecks(model, reduction=reduction)

    # 3. replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    return model


# ── Parameter count utility ────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(num_classes=100)
    total = count_parameters(model)
    print(f"Trainable parameters: {total / 1e6:.2f} M")
    assert total < 100e6, "Model exceeds 100 M parameter limit!"

    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")   # expect (4, 100)

    # verify SE modules are present
    se_count = sum(1 for m in model.modules() if isinstance(m, SEModule))
    print(f"SE modules inserted: {se_count}")   # expect 33 for ResNet-101