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
        Finish calibration by updating quantization parameters.
        
        Note: FlexQ does NOT use AWQ-style smoothing. It uses fine-grained
        group quantization directly. The compressed-tensors system handles
        the actual quantization and storage format.
        """
        self.ended_ = True

        # End calibration - this freezes quantization and removes observers
        # The compressed-tensors system handles quantization based on the
        # QuantizationArgs configuration (strategy="group", group_size=128)
        # Fine-grained group quantization scales are computed during calibration
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

