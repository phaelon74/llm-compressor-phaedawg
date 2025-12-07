"""
Model architecture mappings for FlexQ quantization.

FlexQ uses similar layer mappings to AWQ for identifying
sensitive layers that benefit from higher precision activations.
"""

from dataclasses import dataclass

from loguru import logger
from torch.nn import Module

from llmcompressor.modifiers.awq.mappings import (
    AWQ_MAPPING_REGISTRY,
    get_layer_mappings_from_architecture as get_awq_mappings,
)

__all__ = ["FlexQMapping", "FLEXQ_MAPPING_REGISTRY", "get_layer_mappings_from_architecture"]


@dataclass
class FlexQMapping:
    """
    Dataclass storing config for FlexQ layer mappings.
    FlexQ uses similar structure to AWQ for identifying
    layer relationships, but focuses on sensitivity analysis
    for selective activation quantization.
    """

    # For now, reuse AWQ mappings structure
    # FlexQ can use the same layer relationships
    smooth_layer: str
    balance_layers: list[str]


# FlexQ can reuse AWQ mappings since they identify similar layer relationships
# The difference is in how FlexQ uses these mappings for sensitivity analysis
FLEXQ_MAPPING_REGISTRY = AWQ_MAPPING_REGISTRY


def get_layer_mappings_from_architecture(architecture: str) -> list[FlexQMapping]:
    """
    Get FlexQ layer mappings for a given architecture.
    Currently reuses AWQ mappings since they identify similar structures.

    :param architecture: The architecture name of the model
    :return: List of FlexQ mappings for the architecture
    """
    awq_mappings = get_awq_mappings(architecture)
    
    # Convert AWQ mappings to FlexQ mappings
    flexq_mappings = [
        FlexQMapping(
            smooth_layer=mapping.smooth_layer,
            balance_layers=mapping.balance_layers,
        )
        for mapping in awq_mappings
    ]
    
    return flexq_mappings

