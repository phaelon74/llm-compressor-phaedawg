from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

# =========================
# Model (GLM / GLM-MoE)
# =========================
MODEL_ID = "/home/akitzke/models/TheDrummer/GLM-Steam-106B-A12B-v1/"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# =========================
# Calibration data (WikiText)
# =========================
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

DATASET_ID = "wikitext"
DATASET_NAME = "wikitext-2-raw-v1"
DATASET_SPLIT = "validation"

ds = load_dataset(DATASET_ID, DATASET_NAME, split=DATASET_SPLIT)
ds = ds.filter(lambda ex: ex.get("text", "").strip() != "")

n = min(NUM_CALIBRATION_SAMPLES, len(ds))
ds = ds.shuffle(seed=42).select(range(n))

# Render to chat-style text (batch)
def preprocess(batch):
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
        )
        for t in batch["text"]
    ]
    return {"text": rendered}

ds = ds.map(preprocess, batched=True, num_proc=4)

# Tokenize in batches
ds = ds.map(
    lambda batch: tokenizer(
        batch["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    ),
    batched=True,
    remove_columns=ds.column_names,
    num_proc=4,
)

# =========================
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - group_size: 64
#  - IMPORTANT: do NOT ignore mlp.gate / gate_up_proj (merged layer)
#  - Keep router and output head unquantized
# =========================
moe_ignores = [
    "lm_head",
    "model.embed_tokens",
    "re:.*router.*",              # keep MoE router in FP
    # DO NOT PUT: "re:.*mlp.gate"  <-- removing this avoids the mixed-scheme issue
]

recipe = [
    AWQModifier(
        targets=["Linear"],        # quantize all Linear layers uniformly
        ignore=moe_ignores,
        config_groups={
            "group_0": {
                # Don't list individual proj regexes; let "Linear" + ignore control it
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,   # W4A16 (symmetric)
                    "strategy": "group",
                    "group_size": 64,    # handles dims like 10944
                    "dynamic": False,
                },
            },
        },
        # Optional mappings can be added, but not required
    ),
]

# =========================
# Quantize + save (writes quantization_config for vLLM)
# =========================
SAVE_DIR = "/home/akitzke/models/TheHouseOfTheDude/GLM-Steam-106B-A12B-v1_W4A16_SYM_G64/"

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    output_dir=SAVE_DIR,
)

# (Optional redundant save)
#model.save_pretrained(SAVE_DIR, save_compressed=True)
#tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)

