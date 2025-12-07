"""
Fine-grained group quantization for FlexQ.

FlexQ uses fine-grained group quantization where weights are
quantized in small groups (e.g., 128 elements) rather than
per-channel or per-tensor, providing better accuracy.
"""

import torch
from torch.nn import Module

__all__ = ["quantize_fine_grained_group", "apply_fine_grained_group_quantization"]


def quantize_fine_grained_group(
    weight: torch.Tensor,
    group_size: int = 128,
    bits: int = 6,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """
    Quantize weights using fine-grained group quantization.

    :param weight: Weight tensor to quantize (shape: [out_features, in_features])
    :param group_size: Size of each quantization group
    :param bits: Number of bits for quantization (6 for FlexQ)
    :param symmetric: Whether to use symmetric quantization
    :return: Tuple of (quantized_weight, scales, zero_points)
    """
    org_shape = weight.shape
    
    # Reshape to groups
    if group_size > 0:
        assert org_shape[-1] % group_size == 0, (
            f"Weight shape {org_shape[-1]} must be divisible by group_size {group_size}"
        )
        # Reshape: [out_features, in_features] -> [out_features, num_groups, group_size]
        num_groups = org_shape[-1] // group_size
        weight_reshaped = weight.view(org_shape[0], num_groups, group_size)
    else:
        # Per-channel quantization (group_size = -1)
        weight_reshaped = weight.unsqueeze(1)  # [out_features, 1, in_features]
        group_size = org_shape[-1]
    
    # Quantize each group independently
    if symmetric:
        # Symmetric quantization
        max_val = weight_reshaped.abs().amax(dim=-1, keepdim=True)  # [out_features, num_groups, 1]
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (bits - 1) - 1
        min_int = -(2 ** (bits - 1))
        scales = max_val / max_int
        zero_points = None
        
        # Quantize
        quantized = torch.clamp(
            torch.round(weight_reshaped / scales),
            min_int,
            max_int
        ) * scales
    else:
        # Asymmetric quantization
        max_val = weight_reshaped.amax(dim=-1, keepdim=True)
        min_val = weight_reshaped.amin(dim=-1, keepdim=True)
        max_int = 2**bits - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zero_points = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
        
        # Quantize
        quantized = (
            torch.clamp(
                torch.round(weight_reshaped / scales) + zero_points,
                min_int,
                max_int
            ) - zero_points
        ) * scales
    
    # Reshape back to original shape
    quantized = quantized.view(org_shape)
    
    # Reshape scales and zero_points for storage
    if group_size > 0:
        scales = scales.view(org_shape[0], -1)  # [out_features, num_groups]
        if zero_points is not None:
            zero_points = zero_points.view(org_shape[0], -1)
    else:
        scales = scales.squeeze(1)  # [out_features]
        if zero_points is not None:
            zero_points = zero_points.squeeze(1)
    
    return quantized, scales, zero_points


def apply_fine_grained_group_quantization(
    module: Module,
    group_size: int = 128,
    bits: int = 6,
    symmetric: bool = True,
) -> None:
    """
    Apply fine-grained group quantization to a Linear module's weights.

    :param module: Linear module to quantize
    :param group_size: Size of each quantization group
    :param bits: Number of bits for quantization
    :param symmetric: Whether to use symmetric quantization
    """
    if not isinstance(module, torch.nn.Linear):
        return
    
    if not hasattr(module, "weight") or module.weight is None:
        return
    
    # Quantize weights
    quantized_weight, scales, zero_points = quantize_fine_grained_group(
        module.weight.data,
        group_size=group_size,
        bits=bits,
        symmetric=symmetric,
    )
    
    # Store quantization parameters
    module.register_buffer("weight_scale", scales)
    if zero_points is not None:
        module.register_buffer("weight_zero_point", zero_points)
    else:
        if hasattr(module, "weight_zero_point"):
            delattr(module, "weight_zero_point")
    
    # Update weight (in practice, this would be handled by compressed-tensors)
    # For now, we just store the quantized version
    module.weight.data = quantized_weight

