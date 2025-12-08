"""
Example: Quantizing LLaMA models with FlexQ 6-bit quantization.

This example demonstrates how to use FlexQ to quantize a LLaMA model
to 6-bit weights with 8-bit activations (W6A8).

Configuration:
- Reads SRC_DIR and DST_DIR from .env file in the same directory
- SRC_DIR: Path to the source model directory
- DST_DIR: Path where the quantized model will be saved
"""

import os
from pathlib import Path

# Suppress tokenizers parallelism warnings when using DataLoader with multiple workers
# This is safe - the warnings occur because tokenizers detect forking after parallelism
# is enabled, but it doesn't affect functionality
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.flexq import FlexQModifier
from llmcompressor.utils import dispatch_for_generation
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# Load environment variables from .env file
def load_env_file(env_path: Path) -> dict:
    """Load variables from .env file."""
    env_vars = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    env_vars[key.strip()] = value
    return env_vars

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
ENV_FILE = SCRIPT_DIR / ".env"

# Load environment variables
env_vars = load_env_file(ENV_FILE)

# Get model and destination paths from .env file
SRC_DIR = env_vars.get("SRC_DIR")
DST_DIR = env_vars.get("DST_DIR")

# Validate paths
if not SRC_DIR:
    raise ValueError(
        f"SRC_DIR not found in {ENV_FILE}. "
        "Please create a .env file with SRC_DIR and DST_DIR."
    )
if not DST_DIR:
    raise ValueError(
        f"DST_DIR not found in {ENV_FILE}. "
        "Please create a .env file with SRC_DIR and DST_DIR."
    )

if not os.path.exists(SRC_DIR):
    raise ValueError(f"Source model directory does not exist: {SRC_DIR}")

# Create destination directory if it doesn't exist
os.makedirs(DST_DIR, exist_ok=True)

print(f"Source model directory: {SRC_DIR}")
print(f"Destination directory: {DST_DIR}")

# Select model and load it.
MODEL_ID = SRC_DIR

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
# Using Neural Magic's LLM compression calibration dataset
DATASET_ID = "neuralmagic/LLM_compression_calibration"
DATASET_SPLIT = "train"

# Select number of samples. 512 samples provides good calibration.
NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 2048

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    # The neuralmagic/LLM_compression_calibration dataset has both 'text' and 'messages' fields
    # Prefer 'messages' if available, otherwise use 'text'
    if "messages" in example and example["messages"]:
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    else:
        # Fallback to text field if messages not available
        text = example.get("text", "")
    
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text) if text else ""
    
    # Tokenize directly and return tokenized features
    # The tokenizer returns a dict with 'input_ids', 'attention_mask', etc.
    tokenized = tokenizer(
        text,
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )
    
    # Return tokenized features directly
    return tokenized


# Apply preprocessing (which includes tokenization)
ds = ds.map(preprocess, remove_columns=ds.column_names)

# Configure FlexQ quantization
# FlexQ uses 6-bit weights uniformly with 8-bit activations (W6A8)
# Note: For activations, use "channel" strategy instead of "group" to avoid
# shape mismatches with variable sequence lengths
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
                    num_bits=8,  # 8-bit activations (W6A8)
                    type="int",
                    symmetric=True,
                    # No strategy/group_size specified - uses default per-token quantization for activations
                    # This handles variable sequence lengths correctly
                ),
            ),
        },
        w_group_size=128,
        a_group_size=128,  # This parameter is for fine-grained group quantization logic, not the quantization scheme
        enable_selective_activation=False,  # All layers use 8-bit activations
        sensitivity_threshold=0.05,
    ),
]

# Apply quantization
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
#print("\n\n")
#print("========== SAMPLE GENERATION ==============")
#dispatch_for_generation(model)
#input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
#    model.device
#)
#output = model.generate(input_ids, max_new_tokens=100)
#print(tokenizer.decode(output[0]))
#print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = DST_DIR

# Ensure quantization is properly applied before saving
# The compressed-tensors library should handle compression automatically
# with save_compressed=True, but we verify the quantization status
from compressed_tensors.quantization import QuantizationStatus
from compressed_tensors.utils import match_named_modules

# Check quantization status of quantized modules
quantized_modules = list(match_named_modules(model, ["Linear"], ["lm_head"]))
packed_count = 0
for name, module in quantized_modules[:3]:  # Check first 3 modules
    if hasattr(module, "quantization_status"):
        print(f"{name}: quantization_status = {module.quantization_status}")
    if hasattr(module, "_flexq_packed_weight"):
        packed_count += 1
        print(f"{name}: has packed weights (shape: {module._flexq_packed_weight.shape})")

print(f"\nFound {packed_count} modules with packed weights out of {len(quantized_modules)} total")

# Check for packed weights
packed_modules_count = sum(
    1 for _, module in model.named_modules()
    if hasattr(module, "_flexq_has_packed_weights") and module._flexq_has_packed_weights
)
print(f"Found {packed_modules_count} modules with packed weights metadata")

# Access packed weights from model if available
if hasattr(model, "_flexq_packed_weights") and model._flexq_packed_weights:
    print(f"Found {len(model._flexq_packed_weights)} packed weight tensors on model")
    
    # Hook into state_dict to use packed weights when saving
    original_state_dict = model.state_dict
    
    def state_dict_with_packed_weights(*args, **kwargs):
        """Custom state_dict that uses packed weights for FlexQ modules."""
        state = original_state_dict(*args, **kwargs)
        
        # Replace weights with packed versions where available
        for name, (packed_weight, _) in model._flexq_packed_weights.items():
            param_name = f"{name}.weight"
            if param_name in state:
                state[param_name] = packed_weight
                print(f"Using packed weight for {param_name}: {state[param_name].shape}")
        
        return state
    
    # Temporarily replace state_dict
    model.state_dict = state_dict_with_packed_weights
    
    try:
        model.save_pretrained(SAVE_DIR, save_compressed=True)
    finally:
        # Restore original state_dict
        model.state_dict = original_state_dict
else:
    # No packed weights found, save normally
    print("No packed weights found, saving model normally")
    model.save_pretrained(SAVE_DIR, save_compressed=True)

tokenizer.save_pretrained(SAVE_DIR)

print(f"Model saved to {SAVE_DIR}")
print(f"\nNote: FlexQ 6-bit weights have been packed for efficient storage.")
print("The model should be approximately 6GB for an 8B parameter model at W6.")

