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
    if "messages" in example:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
            )
        }
    else:
        # Fallback to text field if messages not available
        return {"text": example.get("text", "")}


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize)

# Configure FlexQ quantization
# FlexQ uses 6-bit weights uniformly with 8-bit activations (W6A8)
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
                    strategy="group",
                    group_size=128,
                ),
            ),
        },
        w_group_size=128,
        a_group_size=128,
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
print("\n\n")
print("========== SAMPLE GENERATION ==============")
dispatch_for_generation(model)
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
    model.device
)
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")

# Save to disk compressed.
SAVE_DIR = DST_DIR
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model saved to {SAVE_DIR}")

