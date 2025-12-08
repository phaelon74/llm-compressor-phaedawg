"""
6-bit weight packing for FlexQ quantization.

FlexQ uses bit-level data packing to efficiently store 6-bit quantized weights.
This module implements the packing logic to convert quantized INT6 weights
into a packed format for efficient storage and inference.
"""

import torch
from loguru import logger

__all__ = ["pack_int6_weights", "unpack_int6_weights"]


def pack_int6_weights(weight: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor | None = None, group_size: int = 128) -> tuple[torch.Tensor, dict]:
    """
    Pack 6-bit quantized weights into a dense format.
    
    Packing strategy: 4 INT6 values (24 bits) fit exactly into 3 bytes.
    This provides efficient storage while maintaining byte alignment.
    
    Note: The input weight tensor is typically FP16/FP32 with quantization
    metadata. We quantize to INT6 and then pack.
    
    :param weight: Weight tensor (FP16/FP32 format)
    :param scales: Quantization scales tensor (shape: [out_features, num_groups] for group quantization)
    :param zero_points: Quantization zero points tensor (None for symmetric quantization)
    :param group_size: Group size for quantization (default: 128)
    :return: Tuple of (packed_weights, metadata_dict)
    """
    original_shape = weight.shape
    assert len(original_shape) == 2, f"Expected 2D weight tensor, got shape {original_shape}"
    out_features, in_features = original_shape
    
    # Quantize weights to INT6
    if zero_points is None:
        # Symmetric quantization: range is [-32, 31]
        max_int = 2 ** (6 - 1) - 1  # 31
        min_int = -(2 ** (6 - 1))   # -32
        
        # Handle group-wise quantization
        if scales.dim() == 2:
            # Group quantization: scales shape is [out_features, num_groups]
            num_groups = scales.shape[-1]
            assert in_features % num_groups == 0, f"in_features ({in_features}) must be divisible by num_groups ({num_groups})"
            group_size_actual = in_features // num_groups
            
            # Reshape weight to [out_features, num_groups, group_size_actual]
            weight_reshaped = weight.view(out_features, num_groups, group_size_actual)
            scales_expanded = scales.unsqueeze(-1)  # [out_features, num_groups, 1]
            
            # Quantize each group
            quantized = torch.clamp(
                torch.round(weight_reshaped / scales_expanded),
                min_int,
                max_int
            ).int()
            
            # Reshape back
            quantized = quantized.view(original_shape)
        else:
            # Per-channel quantization
            scales_expanded = scales.view(-1, 1) if scales.dim() == 1 else scales
            quantized = torch.clamp(
                torch.round(weight / scales_expanded),
                min_int,
                max_int
            ).int()
        
        # Convert to unsigned for packing: [-32, 31] -> [0, 63]
        weight_unsigned = quantized + 32
    else:
        # Asymmetric quantization: range is [0, 63]
        max_int = 2 ** 6 - 1  # 63
        min_int = 0
        
        # Handle group-wise quantization
        if scales.dim() == 2:
            num_groups = scales.shape[-1]
            assert in_features % num_groups == 0
            group_size_actual = in_features // num_groups
            
            weight_reshaped = weight.view(out_features, num_groups, group_size_actual)
            scales_expanded = scales.unsqueeze(-1)
            zero_points_expanded = zero_points.unsqueeze(-1)
            
            quantized = torch.clamp(
                torch.round(weight_reshaped / scales_expanded) + zero_points_expanded,
                min_int,
                max_int
            ).int()
            
            quantized = quantized.view(original_shape)
        else:
            scales_expanded = scales.view(-1, 1) if scales.dim() == 1 else scales
            zero_points_expanded = zero_points.view(-1, 1) if zero_points.dim() == 1 else zero_points
            quantized = torch.clamp(
                torch.round(weight / scales_expanded) + zero_points_expanded,
                min_int,
                max_int
            ).int()
        
        weight_unsigned = quantized
    
    weight_unsigned = weight_unsigned.clamp(0, 63)
    
    # Flatten the weight tensor for packing
    original_shape = weight_unsigned.shape
    weight_flat = weight_unsigned.flatten()
    num_elements = weight_flat.numel()
    
    # Pad to multiple of 4 for efficient packing (4 values per 3 bytes)
    pad_size = (4 - (num_elements % 4)) % 4
    if pad_size > 0:
        weight_padded = torch.cat([weight_flat, torch.zeros(pad_size, dtype=torch.int32, device=weight.device)])
    else:
        weight_padded = weight_flat
    
    # Reshape to groups of 4
    num_groups = len(weight_padded) // 4
    weight_groups = weight_padded[:num_groups * 4].view(num_groups, 4)
    
    # Pack 4 INT6 values (each 6 bits) into 3 bytes (24 bits total)
    # Each group of 4 values becomes 3 bytes
    packed = torch.zeros(num_groups, 3, dtype=torch.uint8, device=weight.device)
    
    # Pack each group: [v0, v1, v2, v3] -> [byte0, byte1, byte2]
    # byte0 = v0[5:0] (6 bits) + v1[1:0] (2 bits) = v0 | (v1 << 6)
    # byte1 = v1[5:2] (4 bits) + v2[3:0] (4 bits) = (v1 >> 2) | (v2 << 4)
    # byte2 = v2[5:4] (2 bits) + v3[5:0] (6 bits) = (v2 >> 4) | (v3 << 2)
    
    v0 = weight_groups[:, 0].int()
    v1 = weight_groups[:, 1].int()
    v2 = weight_groups[:, 2].int()
    v3 = weight_groups[:, 3].int()
    
    packed[:, 0] = (v0 | (v1 << 6)).clamp(0, 255).to(torch.uint8)
    packed[:, 1] = ((v1 >> 2) | (v2 << 4)).clamp(0, 255).to(torch.uint8)
    packed[:, 2] = ((v2 >> 4) | (v3 << 2)).clamp(0, 255).to(torch.uint8)
    
    # Store metadata for unpacking
    metadata = {
        "original_shape": original_shape,
        "num_elements": num_elements,
        "pad_size": pad_size,
        "symmetric": zero_points is None,
    }
    
    logger.debug(
        f"Packed {num_elements} INT6 values into {packed.numel()} bytes "
        f"(original would be {num_elements} bytes if stored as INT8)"
    )
    
    return packed, metadata


def unpack_int6_weights(packed: torch.Tensor, metadata: dict) -> torch.Tensor:
    """
    Unpack 6-bit quantized weights from packed format.
    
    :param packed: Packed weight tensor (3 bytes per 4 INT6 values)
    :param metadata: Metadata dictionary from packing
    :return: Unpacked weight tensor (INT6 values as INT32)
    """
    num_groups = packed.shape[0]
    
    # Unpack each group of 3 bytes back to 4 INT6 values
    byte0 = packed[:, 0].int()
    byte1 = packed[:, 1].int()
    byte2 = packed[:, 2].int()
    
    # Extract values using bit masks
    v0 = byte0 & 0x3F  # 6 bits: 0b00111111
    v1 = ((byte0 >> 6) & 0x03) | ((byte1 & 0x0F) << 2)  # 6 bits
    v2 = ((byte1 >> 4) & 0x0F) | ((byte2 & 0x03) << 4)  # 6 bits
    v3 = (byte2 >> 2) & 0x3F  # 6 bits
    
    # Combine values
    unpacked = torch.stack([v0, v1, v2, v3], dim=1).flatten()
    
    # Remove padding
    num_elements = metadata["num_elements"]
    unpacked = unpacked[:num_elements]
    
    # Convert back to signed if symmetric
    if metadata["symmetric"]:
        unpacked = unpacked - 32
    
    # Reshape to original shape
    unpacked = unpacked.view(metadata["original_shape"])
    
    return unpacked

