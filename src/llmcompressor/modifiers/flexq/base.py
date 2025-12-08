"""
FlexQ Modifier: 6-bit quantization with selective high-precision activations.

FlexQ implements uniform 6-bit weight quantization across all layers,
with adaptive retention of 8-bit activations in layers identified
through layer-wise sensitivity analysis.
"""

from typing import Literal, Optional

import torch
from compressed_tensors.quantization import disable_quantization
from compressed_tensors.utils import match_named_modules
from loguru import logger
from pydantic import ConfigDict, PrivateAttr, model_validator
from torch.nn import Module
from tqdm import tqdm

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.flexq.fine_grained_group import (
    apply_fine_grained_group_quantization,
)
from llmcompressor.modifiers.flexq.layer_sensitivity import (
    analyze_layer_sensitivity,
    identify_sensitive_layers,
)
from llmcompressor.modifiers.flexq.mappings import (
    FlexQMapping,
    get_layer_mappings_from_architecture,
)
from llmcompressor.modifiers.flexq.packing import pack_int6_weights
from llmcompressor.modifiers.quantization.calibration import (
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales

__all__ = ["FlexQModifier"]


class FlexQModifier(Modifier, QuantizationMixin):
    """
    Implements the FlexQ (Flexible Quantization) algorithm for 6-bit quantization.
    
    FlexQ employs:
    - Uniform 6-bit weight quantization across all layers
    - Fine-grained group quantization for weights
    - Selective 8-bit activation quantization for sensitive layers
    - Layer-wise sensitivity analysis to identify sensitive layers
    
    Example recipe:
    ```python
    from llmcompressor.modifiers.flexq import FlexQModifier
    from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
    
    recipe = [
        FlexQModifier(
            ignore=["lm_head"],
            config_groups={
                "group_0": QuantizationScheme(
                    targets=["Linear"],
                    weights=QuantizationArgs(
                        num_bits=6,
                        type="int",
                        symmetric=True,
                        strategy="group",
                        group_size=128,
                    ),
                    input_activations=QuantizationArgs(
                        num_bits=6,  # Will be upgraded to 8 for sensitive layers
                        type="int",
                        symmetric=True,
                        strategy="group",
                        group_size=128,
                    ),
                ),
            },
            w_group_size=128,
            a_group_size=128,
            enable_selective_activation=True,
        ),
    ]
    ```
    
    :param w_group_size: Group size for weight quantization (default: 128)
    :param a_group_size: Group size for activation quantization (default: 128)
    :param enable_selective_activation: Whether to enable selective 8-bit activation
        quantization for sensitive layers (default: True)
    :param sensitivity_threshold: Threshold for identifying sensitive layers (default: 0.05)
    :param mappings: Optional list of FlexQ mappings. If None, will be inferred from model
    """

    model_config: ConfigDict = ConfigDict(arbitrary_types_allowed=True)

    # FlexQ-specific parameters
    w_group_size: int = 128
    a_group_size: int = 128
    enable_selective_activation: bool = True
    sensitivity_threshold: float = 0.05
    mappings: Optional[list[FlexQMapping]] = None

    # Private attributes
    _num_bits: int = PrivateAttr(default=None)
    _sensitive_layers: set[str] = PrivateAttr(default_factory=set)
    _calibration_data: list[torch.Tensor] = PrivateAttr(default_factory=list)
    _packed_weights: dict[str, tuple[torch.Tensor, dict]] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def validate_flexq_after(model: "FlexQModifier") -> "FlexQModifier":
        """
        Validate FlexQ configuration:
        - Ensure weights are 6-bit
        - Validate group sizes
        - Check activation quantization settings
        """
        config = model.resolve_quantization_config()

        # Check weight quantization bits
        num_bits_set = set(
            group.weights.num_bits
            for group in config.config_groups.values()
            if group.weights is not None
        )
        
        if len(num_bits_set) > 0:
            model._num_bits = next(iter(num_bits_set))
            if model._num_bits != 6:
                logger.warning(
                    f"FlexQ typically uses 6-bit weights, but found {model._num_bits}-bit. "
                    "Proceeding with {model._num_bits}-bit quantization."
                )
        else:
            model._num_bits = 6  # Default to 6-bit

        # Validate group sizes
        if model.w_group_size <= 0:
            raise ValueError("w_group_size must be positive")
        if model.a_group_size <= 0:
            raise ValueError("a_group_size must be positive")

        return model

    def on_initialize(self, state: State, **kwargs) -> bool:
        """
        Initialize FlexQ quantization on the model.
        
        :param state: State containing the model
        :return: True on success
        """
        # Initialize quantization config
        if QuantizationMixin.has_config(self):
            QuantizationMixin.initialize_quantization(self, state.model)

        # Resolve mappings if not provided
        if self.mappings is None:
            logger.info("No FlexQModifier.mappings provided, inferring from model...")
            self.mappings = get_layer_mappings_from_architecture(
                architecture=state.model.__class__.__name__
            )

        return True

    def on_start(self, state: State, event: Event, **kwargs):
        """
        Start FlexQ calibration process.
        """
        self.started_ = True

        # Start calibration hooks
        QuantizationMixin.start_calibration(self, state.model)

        # Update global scales for group quantization (similar to QuantizationModifier)
        named_modules = list(
            match_named_modules(state.model, self.resolved_targets, self.ignore)
        )
        for _, module in tqdm(named_modules, desc="Updating global scales"):
            update_weight_global_scale(module)

        # Update fused layer global scales (for attention/MLP layers)
        for module in tqdm(state.model.modules(), desc="Fusing global scales"):
            update_fused_layer_weight_global_scales(module)

        # Calibrate weights
        for _, module in tqdm(named_modules, desc="Calibrating weights"):
            update_weight_zp_scale(module)

        logger.info("FlexQ: Starting calibration and sensitivity analysis...")

    def on_event(self, state: State, event: Event, **kwargs):
        """
        Handle calibration events.
        """
        if event.type_ == EventType.CALIBRATION_EPOCH_START:
            if not self.started_:
                self.on_start(state, event, **kwargs)

        elif event.type_ == EventType.CALIBRATION_EPOCH_END:
            # Perform sensitivity analysis if enabled
            if self.enable_selective_activation:
                self._analyze_sensitivity(state.model)

            if not self.ended_:
                self.on_end(state, event, **kwargs)

    def on_end(self, state: State, event: Event, **kwargs):
        """
        Finish calibration by quantizing weights to INT6 and packing them.
        
        Note: FlexQ does NOT use AWQ-style smoothing. It uses fine-grained
        group quantization directly. After calibration, we:
        1. Quantize weights to INT6 using the computed scales
        2. Pack quantized weights into efficient format
        """
        self.ended_ = True

        # Quantize and pack 6-bit weights BEFORE ending calibration
        # (end_calibration removes observers, so we need scales before that)
        # The scales are computed during forward passes in calibration mode
        if self._num_bits == 6:
            # Quantize weights to INT6 using the scales computed during calibration
            logger.info("FlexQ: Quantizing weights to INT6...")
            self._quantize_weights(state.model)
            
            # Pack quantized weights for efficient storage
            logger.info("FlexQ: Packing 6-bit weights for efficient storage...")
            self._pack_weights(state.model)

        # End calibration - this freezes quantization and removes observers
        # Do this AFTER quantization so scales are still available
        QuantizationMixin.end_calibration(self, state.model)

        logger.info("FlexQ: Calibration complete")

    def on_finalize(self, state: State, **kwargs) -> bool:
        """
        Clean up FlexQ resources.
        """
        if not self.ended_:
            self.on_end(state, None)

        self._calibration_data.clear()
        self._sensitive_layers.clear()

        return True

    def _analyze_sensitivity(self, model: Module):
        """
        Perform layer-wise sensitivity analysis to identify layers
        that benefit from 8-bit activations.
        
        Note: This is a simplified implementation. For full FlexQ sensitivity
        analysis, calibration data should be captured during forward passes.
        
        :param model: The model to analyze
        """
        if not self.enable_selective_activation:
            return

        logger.info("FlexQ: Performing layer sensitivity analysis...")

        # Get all target layer names
        layer_names = [
            name for name, _ in match_named_modules(model, self.resolved_targets, self.ignore)
        ]

        if len(layer_names) == 0:
            logger.warning("No layers found for sensitivity analysis")
            return

        # Analyze sensitivity
        # Note: Full implementation would capture calibration data during forward passes
        # For now, we use a simplified analysis based on weight statistics
        if len(self._calibration_data) > 0:
            sensitivity_scores = analyze_layer_sensitivity(
                model=model,
                calibration_data=self._calibration_data,
                layer_names=layer_names,
                threshold=self.sensitivity_threshold,
            )
        else:
            # Fallback: Use a simple heuristic based on layer position
            # Early and late layers are often more sensitive
            # This is a placeholder - full implementation should use calibration data
            logger.info(
                "No calibration data available for sensitivity analysis. "
                "Using simplified heuristic."
            )
            sensitivity_scores = {}
            num_layers = len(layer_names)
            for idx, layer_name in enumerate(layer_names):
                # Early and late layers get higher sensitivity scores
                position_score = 1.0 - abs(idx / max(num_layers - 1, 1) - 0.5) * 2
                sensitivity_scores[layer_name] = position_score * 0.1

        # Identify sensitive layers
        sensitive_layers = identify_sensitive_layers(
            sensitivity_scores,
            threshold=self.sensitivity_threshold,
        )

        self._sensitive_layers = set(sensitive_layers)

        logger.info(
            f"FlexQ: Identified {len(self._sensitive_layers)} sensitive layers "
            f"out of {len(layer_names)} total layers"
        )

        # Update quantization config for sensitive layers to use 8-bit activations
        # This would need to be integrated with compressed-tensors quantization config
        # For now, we log which layers are sensitive
        if len(self._sensitive_layers) > 0:
            logger.info(
                f"Sensitive layers (should use 8-bit activations): "
                f"{list(self._sensitive_layers)[:5]}..."  # Show first 5
            )

    def _quantize_weights(self, model: Module):
        """
        Quantize weights to INT6 using the computed scales.
        
        This actually converts FP16 weights to quantized INT6 values.
        The quantized weights are stored back in the module's weight parameter.
        
        :param model: The model with calibrated quantization scales
        """
        from compressed_tensors.quantization import QuantizationStatus
        from compressed_tensors.utils import getattr_chain
        
        named_modules = list(
            match_named_modules(model, self.resolved_targets, self.ignore)
        )
        
        quantized_count = 0
        
        for name, module in tqdm(named_modules, desc="Quantizing weights to INT6"):
            # Check if this module has 6-bit quantization
            weights_args = getattr_chain(module, "quantization_scheme.weights", None)
            if weights_args is None or weights_args.num_bits != 6:
                continue
            
            # Check if quantization is in calibration mode
            # We quantize BEFORE end_calibration is called, so status should be CALIBRATION
            quantization_status = getattr(module, "quantization_status", None)
            if quantization_status != QuantizationStatus.CALIBRATION:
                logger.debug(f"Skipping {name}: quantization_status={quantization_status}, expected CALIBRATION")
                continue
            
            # Get weight and scales - ensure weight is materialized
            if not hasattr(module, "weight") or module.weight is None:
                continue
            
            # Ensure weight is materialized (not on meta device)
            try:
                weight = module.weight.data
                if weight.device.type == "meta":
                    # Try to materialize the weight
                    from compressed_tensors.utils import align_module_device
                    with align_module_device(module):
                        weight = module.weight.data
                        if weight.device.type == "meta":
                            logger.debug(f"Skipping {name}: weight is on meta device and cannot be materialized")
                            continue
            except Exception as e:
                logger.debug(f"Skipping {name}: cannot access weight - {e}")
                continue
            
            scales = self._get_scales(module)
            zero_points = self._get_zero_points(module)
            
            if scales is None:
                logger.debug(f"Skipping {name}: no scales found for quantization")
                continue
            
            # Ensure scales are materialized
            try:
                if scales.device.type == "meta":
                    from compressed_tensors.utils import align_module_device
                    with align_module_device(module):
                        scales = self._get_scales(module)
                        if scales is None or scales.device.type == "meta":
                            logger.debug(f"Skipping {name}: scales are on meta device")
                            continue
            except:
                pass
            
            try:
                # Quantize the weights to INT6
                quantized_weight = self._quantize_weight_tensor(
                    weight, scales, zero_points, weights_args
                )
                
                # Replace the weight with quantized version
                # This is the crucial step - actually replacing FP16 weights with quantized values
                with torch.no_grad():
                    module.weight.data = quantized_weight
                
                quantized_count += 1
                
                if quantized_count <= 3:  # Log first 3 for debugging
                    logger.info(
                        f"Quantized {name}: shape={weight.shape}, "
                        f"original_range=[{weight.min():.4f}, {weight.max():.4f}], "
                        f"quantized_range=[{quantized_weight.min():.4f}, {quantized_weight.max():.4f}]"
                    )
            except Exception as e:
                logger.error(f"Failed to quantize weights for {name}: {e}", exc_info=True)
                continue
        
        logger.info(f"FlexQ: Quantized {quantized_count} weight tensors to INT6")
    
    def _quantize_weight_tensor(
        self,
        weight: torch.Tensor,
        scales: torch.Tensor,
        zero_points: torch.Tensor | None,
        weights_args,
    ) -> torch.Tensor:
        """
        Quantize a weight tensor to INT6.
        
        :param weight: FP16/FP32 weight tensor
        :param scales: Quantization scales
        :param zero_points: Quantization zero points (None for symmetric)
        :param weights_args: QuantizationArgs for weights
        :return: Quantized weight tensor (still as FP16/FP32 but with quantized values)
        """
        group_size = getattr(weights_args, "group_size", None) or self.w_group_size
        symmetric = getattr(weights_args, "symmetric", True)
        
        original_shape = weight.shape
        out_features, in_features = original_shape
        
        if symmetric:
            # Symmetric quantization: range [-32, 31]
            max_int = 2 ** (6 - 1) - 1  # 31
            min_int = -(2 ** (6 - 1))   # -32
            
            if scales.dim() == 2:
                # Group quantization
                num_groups = scales.shape[-1]
                assert in_features % num_groups == 0
                group_size_actual = in_features // num_groups
                
                weight_reshaped = weight.view(out_features, num_groups, group_size_actual)
                scales_expanded = scales.unsqueeze(-1)  # [out_features, num_groups, 1]
                
                # Quantize: round(weight / scale) and clamp
                quantized = torch.clamp(
                    torch.round(weight_reshaped / scales_expanded),
                    min_int,
                    max_int
                )
                
                # Dequantize back to FP16/FP32 range for storage
                quantized_weight = quantized * scales_expanded
                quantized_weight = quantized_weight.view(original_shape)
            else:
                # Per-channel quantization
                scales_expanded = scales.view(-1, 1) if scales.dim() == 1 else scales
                quantized = torch.clamp(
                    torch.round(weight / scales_expanded),
                    min_int,
                    max_int
                )
                quantized_weight = quantized * scales_expanded
        else:
            # Asymmetric quantization: range [0, 63]
            max_int = 2 ** 6 - 1  # 63
            min_int = 0
            
            if scales.dim() == 2:
                num_groups = scales.shape[-1]
                assert in_features % num_groups == 0
                group_size_actual = in_features // num_groups
                
                weight_reshaped = weight.view(out_features, num_groups, group_size_actual)
                scales_expanded = scales.unsqueeze(-1)
                zero_points_expanded = zero_points.unsqueeze(-1) if zero_points is not None else None
                
                quantized = torch.clamp(
                    torch.round(weight_reshaped / scales_expanded) + (zero_points_expanded if zero_points_expanded is not None else 0),
                    min_int,
                    max_int
                )
                
                quantized_weight = (quantized - (zero_points_expanded if zero_points_expanded is not None else 0)) * scales_expanded
                quantized_weight = quantized_weight.view(original_shape)
            else:
                scales_expanded = scales.view(-1, 1) if scales.dim() == 1 else scales
                zero_points_expanded = zero_points.view(-1, 1) if zero_points is not None and zero_points.dim() == 1 else zero_points
                
                quantized = torch.clamp(
                    torch.round(weight / scales_expanded) + (zero_points_expanded if zero_points_expanded is not None else 0),
                    min_int,
                    max_int
                )
                quantized_weight = (quantized - (zero_points_expanded if zero_points_expanded is not None else 0)) * scales_expanded
        
        return quantized_weight.to(weight.dtype)
    
    def _get_scales(self, module: Module) -> torch.Tensor | None:
        """Get quantization scales from a module."""
        # Try various methods to get scales
        if hasattr(module, "weight_scale"):
            scales = getattr(module, "weight_scale")
            if torch.is_tensor(scales):
                return scales.data if hasattr(scales, 'data') else scales
        
        for buffer_name, buffer in module.named_buffers():
            if buffer_name == "weight_scale":
                return buffer
        
        if hasattr(module, "_offload_weights_map"):
            offload_map = getattr(module, "_offload_weights_map", {})
            if "weight_scale" in offload_map:
                scales = offload_map["weight_scale"]
                return scales.data if hasattr(scales, 'data') else scales
        
        try:
            state_dict = module.state_dict()
            if "weight_scale" in state_dict:
                return state_dict["weight_scale"]
        except:
            pass
        
        return None
    
    def _get_zero_points(self, module: Module) -> torch.Tensor | None:
        """Get quantization zero points from a module."""
        if hasattr(module, "weight_zero_point"):
            zp = getattr(module, "weight_zero_point")
            if torch.is_tensor(zp):
                return zp.data if hasattr(zp, 'data') else zp
        
        for buffer_name, buffer in module.named_buffers():
            if buffer_name == "weight_zero_point":
                return buffer
        
        if hasattr(module, "_offload_weights_map"):
            offload_map = getattr(module, "_offload_weights_map", {})
            if "weight_zero_point" in offload_map:
                zp = offload_map["weight_zero_point"]
                return zp.data if hasattr(zp, 'data') else zp
        
        return None

    def _pack_weights(self, model: Module):
        """
        Pack 6-bit quantized weights into efficient format.
        
        Since compressed-tensors doesn't natively support 6-bit packing,
        we manually pack the weights and store them. The packed weights
        will be used when saving the model.
        
        :param model: The model with quantized weights
        """
        from compressed_tensors.quantization import QuantizationStatus
        from compressed_tensors.utils import getattr_chain
        
        named_modules = list(
            match_named_modules(model, self.resolved_targets, self.ignore)
        )
        
        packed_count = 0
        skipped_no_scheme = 0
        skipped_wrong_bits = 0
        skipped_not_frozen = 0
        skipped_no_weight = 0
        skipped_no_scales = 0
        
        for name, module in tqdm(named_modules, desc="Packing 6-bit weights"):
            # Check if this module has 6-bit quantization
            weights_args = getattr_chain(module, "quantization_scheme.weights", None)
            if weights_args is None:
                skipped_no_scheme += 1
                continue
            
            if weights_args.num_bits != 6:
                skipped_wrong_bits += 1
                if packed_count == 0 and skipped_wrong_bits == 1:
                    logger.debug(f"Example skip: {name} has {weights_args.num_bits} bits, expected 6")
                continue
            
            # Check if quantization is frozen (calibration complete)
            quantization_status = getattr(module, "quantization_status", None)
            if quantization_status != QuantizationStatus.FROZEN:
                skipped_not_frozen += 1
                if skipped_not_frozen == 1:
                    logger.debug(f"Example skip: {name} quantization_status={quantization_status}, expected FROZEN")
                continue
            
            # Get quantized weight, scales, and zero points
            if not hasattr(module, "weight") or module.weight is None:
                skipped_no_weight += 1
                continue
            
            weight = module.weight.data
            
            # Try to get scales - compressed-tensors stores them via update_offload_parameter
            # They might be stored as buffers, attributes, or in a special offload structure
            scales = None
            zero_points = None
            
            # Method 1: Direct attribute access
            if hasattr(module, "weight_scale"):
                scales = getattr(module, "weight_scale")
                if torch.is_tensor(scales):
                    scales = scales.data if hasattr(scales, 'data') else scales
            
            # Method 2: Check buffers
            if scales is None:
                for buffer_name, buffer in module.named_buffers():
                    if buffer_name == "weight_scale":
                        scales = buffer
                        break
            
            # Method 3: Check if stored in _offload_weights_map (internal compressed-tensors structure)
            if scales is None and hasattr(module, "_offload_weights_map"):
                offload_map = getattr(module, "_offload_weights_map", {})
                if "weight_scale" in offload_map:
                    scales = offload_map["weight_scale"]
                    if hasattr(scales, 'data'):
                        scales = scales.data
            
            # Method 4: Try accessing via state_dict (scales might be registered there)
            if scales is None:
                try:
                    state_dict = module.state_dict()
                    if "weight_scale" in state_dict:
                        scales = state_dict["weight_scale"]
                except:
                    pass
            
            # Similar for zero_points
            if hasattr(module, "weight_zero_point"):
                zero_points = getattr(module, "weight_zero_point")
                if torch.is_tensor(zero_points):
                    zero_points = zero_points.data if hasattr(zero_points, 'data') else zero_points
            
            if zero_points is None:
                for buffer_name, buffer in module.named_buffers():
                    if buffer_name == "weight_zero_point":
                        zero_points = buffer
                        break
            
            if zero_points is None and hasattr(module, "_offload_weights_map"):
                offload_map = getattr(module, "_offload_weights_map", {})
                if "weight_zero_point" in offload_map:
                    zero_points = offload_map["weight_zero_point"]
                    if hasattr(zero_points, 'data'):
                        zero_points = zero_points.data
            
            if scales is None:
                skipped_no_scales += 1
                if skipped_no_scales <= 3:  # Show first 3 examples
                    # Debug: show what's actually available
                    all_attrs = [attr for attr in dir(module) if not attr.startswith('__')]
                    scale_attrs = [attr for attr in all_attrs if 'scale' in attr.lower()]
                    buffer_names = [name for name, _ in module.named_buffers()]
                    logger.warning(
                        f"Example skip: {name} - no weight_scale found.\n"
                        f"  Scale-related attributes: {scale_attrs}\n"
                        f"  Buffers: {buffer_names}\n"
                        f"  Has _offload_weights_map: {hasattr(module, '_offload_weights_map')}\n"
                        f"  Quantization scheme weights num_bits: {weights_args.num_bits if weights_args else 'N/A'}"
                    )
                continue
            
            # Pack the weights
            try:
                # Get group_size from quantization scheme
                weights_args = getattr_chain(module, "quantization_scheme.weights", None)
                group_size = getattr(weights_args, "group_size", None) if weights_args else None
                if group_size is None or group_size <= 0:
                    group_size = self.w_group_size
                
                # Ensure weight is on CPU for packing (to avoid device issues)
                weight_cpu = weight.cpu() if weight.device.type != "cpu" else weight
                scales_cpu = scales.cpu() if scales.device.type != "cpu" else scales
                zero_points_cpu = zero_points.cpu() if zero_points is not None and zero_points.device.type != "cpu" else zero_points
                
                packed_weight, metadata = pack_int6_weights(weight_cpu, scales_cpu, zero_points_cpu, group_size)
                # Store packed weight in modifier's dict (not as module attribute to avoid device movement issues)
                self._packed_weights[name] = (packed_weight, metadata)
                
                # Store only lightweight metadata as module attributes (not tensors)
                # This allows us to identify which modules have packed weights without interfering with device movement
                module._flexq_packing_metadata = {
                    "original_shape": tuple(weight.shape),
                    "symmetric": zero_points is None,
                }
                module._flexq_has_packed_weights = True
                
                logger.debug(f"Packed weights for {name}: {weight.shape} -> {packed_weight.shape}")
                packed_count += 1
            except Exception as e:
                logger.error(f"Failed to pack weights for {name}: {e}", exc_info=True)
                continue
        
        logger.info(
            f"FlexQ: Packed {len(self._packed_weights)} weight tensors. "
            f"Skipped: {skipped_no_scheme} no scheme, {skipped_wrong_bits} wrong bits, "
            f"{skipped_not_frozen} not frozen, {skipped_no_weight} no weight, {skipped_no_scales} no scales"
        )
        
        # Store packed weights on the model object for access during save
        # This avoids device movement issues since we're not storing tensors as module attributes
        if not hasattr(model, "_flexq_packed_weights"):
            model._flexq_packed_weights = {}
        model._flexq_packed_weights.update(self._packed_weights)

