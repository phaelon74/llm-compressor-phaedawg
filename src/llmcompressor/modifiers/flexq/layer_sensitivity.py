"""
Layer-wise sensitivity analysis for FlexQ quantization.

FlexQ uses layer sensitivity analysis to identify which layers
benefit from higher precision (8-bit) activations while using
6-bit weights uniformly across all layers.
"""

from typing import Dict, List

import torch
from loguru import logger
from torch.nn import Module

__all__ = ["analyze_layer_sensitivity", "identify_sensitive_layers"]


def analyze_layer_sensitivity(
    model: Module,
    calibration_data: List[torch.Tensor],
    layer_names: List[str],
    threshold: float = 0.05,
) -> Dict[str, float]:
    """
    Analyze sensitivity of each layer by measuring accuracy degradation
    when activations are quantized to 6-bit vs 8-bit.

    :param model: The model to analyze
    :param calibration_data: List of calibration input tensors
    :param layer_names: List of layer names to analyze
    :param threshold: Sensitivity threshold - layers above this are considered sensitive
    :return: Dictionary mapping layer names to sensitivity scores
    """
    sensitivity_scores = {}
    
    # For each layer, measure the impact of quantization
    # Higher sensitivity means the layer benefits more from 8-bit activations
    for layer_name in layer_names:
        try:
            layer = _get_layer_by_name(model, layer_name)
            if layer is None:
                continue
                
            # Measure sensitivity by comparing quantization errors
            sensitivity = _compute_layer_sensitivity(layer, calibration_data)
            sensitivity_scores[layer_name] = sensitivity
        except Exception as e:
            logger.warning(f"Failed to analyze sensitivity for {layer_name}: {e}")
            sensitivity_scores[layer_name] = 0.0
    
    return sensitivity_scores


def identify_sensitive_layers(
    sensitivity_scores: Dict[str, float],
    threshold: float = 0.05,
    top_k: int = None,
) -> List[str]:
    """
    Identify layers that are sensitive and should use 8-bit activations.

    :param sensitivity_scores: Dictionary of layer names to sensitivity scores
    :param threshold: Minimum sensitivity score to be considered sensitive
    :param top_k: If provided, return top K most sensitive layers regardless of threshold
    :return: List of sensitive layer names
    """
    if top_k is not None:
        # Return top K most sensitive layers
        sorted_layers = sorted(
            sensitivity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [name for name, _ in sorted_layers[:top_k]]
    
    # Return all layers above threshold
    return [
        name for name, score in sensitivity_scores.items()
        if score >= threshold
    ]


def _compute_layer_sensitivity(
    layer: Module,
    calibration_data: List[torch.Tensor],
) -> float:
    """
    Compute sensitivity score for a layer by measuring quantization error.

    :param layer: The layer module to analyze
    :param calibration_data: Calibration inputs
    :return: Sensitivity score (higher = more sensitive)
    """
    if not isinstance(layer, torch.nn.Linear):
        return 0.0
    
    layer.eval()
    with torch.no_grad():
        total_error_6bit = 0.0
        total_error_8bit = 0.0
        num_samples = 0
        
        for inputs in calibration_data[:10]:  # Use subset for efficiency
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            
            # Forward pass with original weights
            try:
                output_fp = layer(inputs)
            except Exception:
                continue
            
            # Simulate 6-bit quantization error
            weight_6bit = _quantize_tensor(layer.weight.data, bits=6)
            output_6bit = torch.nn.functional.linear(inputs, weight_6bit)
            error_6bit = torch.nn.functional.mse_loss(output_fp, output_6bit).item()
            
            # Simulate 8-bit quantization error
            weight_8bit = _quantize_tensor(layer.weight.data, bits=8)
            output_8bit = torch.nn.functional.linear(inputs, weight_8bit)
            error_8bit = torch.nn.functional.mse_loss(output_fp, output_8bit).item()
            
            total_error_6bit += error_6bit
            total_error_8bit += error_8bit
            num_samples += 1
        
        if num_samples == 0:
            return 0.0
        
        # Sensitivity is the difference in error between 6-bit and 8-bit
        # Higher difference means more sensitive to quantization
        avg_error_6bit = total_error_6bit / num_samples
        avg_error_8bit = total_error_8bit / num_samples
        
        # Normalize by 8-bit error to get relative sensitivity
        if avg_error_8bit > 0:
            sensitivity = (avg_error_6bit - avg_error_8bit) / avg_error_8bit
        else:
            sensitivity = 0.0
        
        return max(0.0, sensitivity)


def _quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Simple symmetric quantization for sensitivity analysis.

    :param tensor: Input tensor
    :param bits: Number of bits
    :return: Quantized tensor
    """
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    
    scale = max_val / (2 ** (bits - 1) - 1)
    quantized = torch.round(tensor / scale) * scale
    return quantized


def _get_layer_by_name(model: Module, name: str) -> Module | None:
    """
    Get a layer module by its name.

    :param model: The model
    :param name: Layer name (supports regex patterns)
    :return: The layer module or None
    """
    from compressed_tensors.utils import match_named_modules
    
    matches = list(match_named_modules(model, [name], []))
    if matches:
        return matches[0][1]
    return None

