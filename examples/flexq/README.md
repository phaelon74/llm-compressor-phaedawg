# Quantizing Models with FlexQ 6-bit Quantization

FlexQ is a novel post-training INT6 quantization framework that combines algorithmic innovation with system-level optimizations. FlexQ employs uniform 6-bit weight quantization across all layers, with adaptive retention of 8-bit activations in layers identified through layer-wise sensitivity analysis.

## Key Features

- **Uniform 6-bit Weight Quantization**: All weights are quantized to 6-bit precision using fine-grained group quantization
- **Selective Activation Quantization**: Sensitive layers automatically use 8-bit activations while others use 6-bit
- **Fine-grained Group Quantization**: Weights are quantized in small groups (e.g., 128 elements) for better accuracy
- **Layer Sensitivity Analysis**: Automatically identifies layers that benefit from higher precision activations

## FlexQ Recipe

The FlexQ recipe uses `config_groups` to specify quantization parameters:

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
                    num_bits=6,  # Upgraded to 8-bit for sensitive layers
                    type="int",
                    symmetric=True,
                    strategy="group",
                    group_size=128,
                ),
            ),
        },
        w_group_size=128,  # Weight quantization group size
        a_group_size=128,  # Activation quantization group size
        enable_selective_activation=True,  # Enable sensitivity analysis
        sensitivity_threshold=0.05,  # Threshold for identifying sensitive layers
    ),
]
```

## Parameters

- `w_group_size`: Group size for weight quantization (default: 128)
- `a_group_size`: Group size for activation quantization (default: 128)
- `enable_selective_activation`: Whether to enable selective 8-bit activation quantization (default: True)
- `sensitivity_threshold`: Threshold for identifying sensitive layers (default: 0.05)
- `mappings`: Optional list of FlexQ mappings. If None, will be inferred from model architecture

## Compressing Your Own Model

To use your own model, start with an existing example and change the `model_id`:

```python
model_id = "path/to/your/model"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
```

## Supported Architectures

FlexQ supports the same architectures as AWQ, including:
- LLaMA family (Llama, Llama2, Llama3, Llama4)
- Mistral
- Qwen family
- Phi
- Gemma
- And more...

## Accuracy

FlexQ maintains near-FP16 accuracy with perplexity increases of no more than 0.1 on WikiText2, while providing significant memory savings and inference acceleration.

## References

- FlexQ Paper: [FlexQ: Efficient Post-training INT6 Quantization for LLM Serving](https://arxiv.org/abs/2508.04405)
- FlexQ Repository: https://github.com/FlyFoxPlayer/FlexQ

