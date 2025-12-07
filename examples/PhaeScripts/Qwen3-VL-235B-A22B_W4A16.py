from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.utils import logging as hf_logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import requests
import importlib.util
from accelerate import init_empty_weights

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from collections import defaultdict, Counter

# =========================
# Load ENV Variables
# =========================
from dotenv import load_dotenv

# Load the .env that sits next to this script (works regardless of where you run it)
load_dotenv(Path(__file__).with_name(".env"))

def require_env(key: str) -> str:
    val = os.getenv(key)
    if not val or not val.strip():
        raise RuntimeError(f"Missing environment variable: {key}")
    return val.strip()

SRC_DIR = require_env("SRC_DIR")
DST_DIR = require_env("DST_DIR")

def get_env_int(key: str, default: int) -> int:
    try:
        v = os.getenv(key)
        return int(v) if v is not None and v.strip() != "" else default
    except Exception:
        return default

def get_env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    v = v.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


# =========================
# Model (Qwen3-VL-235B-A22B-Instruct)
# =========================
hf_logging.set_verbosity_info()
MODEL_ID = require_env("SRC_DIR")

# Preflight: if local dir, ensure weight shards exist to avoid slow random init
if os.path.isdir(MODEL_ID):
    entries = set(os.listdir(MODEL_ID))
    has_index = (
        "model.safetensors.index.json" in entries
        or "pytorch_model.bin.index.json" in entries
    )
    has_any_shard = any(
        (name.endswith(".safetensors") or name.endswith(".bin")) and "model" in name
        for name in entries
    )
    if not (has_index or has_any_shard):
        raise RuntimeError(
            f"SRC_DIR='{MODEL_ID}' does not contain model weight shards. "
            "Set SRC_DIR to the HF repo id (e.g., 'Qwen/Qwen3-VL-235B-A22B-Instruct') "
            "or a local directory with 'model.safetensors' shards. If using HF Hub, set HF_TOKEN and accept the license."
        )

print(f"Loading model from: {MODEL_ID}")
hf_token = os.getenv("HF_TOKEN")
model = AutoModel.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    trust_remote_code=True,
    token=hf_token,
    low_cpu_mem_usage=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
print("Model and tokenizer loaded.")
model.eval()
torch.set_grad_enabled(False)

# =========================
# Intel-style MoE expert conversion (fused -> per-expert Linear)
# =========================
def _download_modeling_qwen3_vl_moe(target_dir: str) -> str:
    url = (
        "https://huggingface.co/Intel/Qwen3-VL-235B-A22B-Instruct-int4-mixed-AutoRound/"
        "raw/main/modeling_qwen3_vl_moe.py"
    )
    dst_path = Path(target_dir) / "modeling_qwen3_vl_moe.py"
    if not dst_path.exists():
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dst_path.write_bytes(resp.content)
    return str(dst_path)

def _load_custom_qwen3_vl_moe_class(modeling_path: str):
    spec = importlib.util.spec_from_file_location("modeling_qwen3_vl_moe", modeling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to create spec for modeling_qwen3_vl_moe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "Qwen3VLMoeForConditionalGeneration"):
        raise RuntimeError("modeling_qwen3_vl_moe.py missing Qwen3VLMoeForConditionalGeneration")
    return module.Qwen3VLMoeForConditionalGeneration

def _convert_fused_experts_to_linear_state_dict(model, tokenizer, convert_dir: str):
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
    state = model.state_dict()
    keys = list(state.keys())
    text_cfg = getattr(config, "text_config", None)
    num_experts = getattr(text_cfg, "num_experts", None)

    # Rewrite gate_up_proj and down_proj expert tensors into per-expert Linear .weight
    rewrites = 0
    for key in list(keys):
        if "expert" in key and ("gate_up_proj" in key or "down_proj" in key):
            tensor = state[key]
            if tensor is None or tensor.ndim != 3:
                continue
            n_experts = tensor.shape[0] if num_experts is None else num_experts
            # new base key name
            if "gate_up_proj" in key:
                base = key.replace("gate_up_proj", "gate_up_projs")
            else:
                base = key.replace("down_proj", "down_projs")
            # strip trailing .weight if present; we will append .i.weight later
            if base.endswith(".weight"):
                base = base[: -len(".weight")]
            for i in range(n_experts):
                new_key = f"{base}.{i}.weight"
                # transpose last two dims to (out, in)
                value = tensor[i, ...].transpose(0, 1).contiguous()
                state[new_key] = value
                rewrites += 1
            # remove old fused tensor
            state[key] = None
            state.pop(key)

    print(f"[CONVERT] Expert Linear rewrites created: {rewrites}")

    # Prepare target modeling file and build empty model to load converted weights
    Path(convert_dir).mkdir(parents=True, exist_ok=True)
    modeling_path = _download_modeling_qwen3_vl_moe(convert_dir)
    Qwen3VLMoeForConditionalGeneration = _load_custom_qwen3_vl_moe_class(modeling_path)

    with init_empty_weights():
        converted_model = Qwen3VLMoeForConditionalGeneration._from_config(config)

    missing, unexpected = converted_model.load_state_dict(state, strict=False)
    print(f"[CONVERT] load_state_dict missing={len(missing)}, unexpected={len(unexpected)}")
    converted_model.to("cpu")
    converted_model.save_pretrained(convert_dir)
    tokenizer.save_pretrained(convert_dir)
    return converted_model

# =========================
# Calibration data (WikiText)
# =========================
NUM_CALIBRATION_SAMPLES = get_env_int("CALIB_SAMPLES", 64)
MAX_SEQUENCE_LENGTH = get_env_int("MAX_SEQ_LEN", 256)
MAP_NUM_PROC = get_env_int("MAP_NUM_PROC", 1)

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

ds = ds.map(preprocess, batched=True, num_proc=MAP_NUM_PROC)

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
    num_proc=MAP_NUM_PROC,
)

# =========================
# AWQ recipe with config_groups
#  - Weight-only INT4 (W4A16 **symmetric**)
#  - group_size: 128
#  - IMPORTANT: do NOT ignore FFN gate/up/down proj (quantize them)
#  - Keep MoE router-related linears, MoE shared expert gate/gate, and output head unquantized
# =========================
router_keywords = (
    "router",
    "expert_choice",
    "dispatch",
    "scores",
    "route",
    "topk",
    "switch",
)

# Exclude VL vision tower and multimodal projector components from quantization
vision_type_markers = (
    "vision",
    "visual",
    "tower",
    "image",
    "pixel",
    "clip",
    "vit",
    "resampler",
    "projector",
    "multimodal",
    "mm",
    "vl",
)

def _build_name_maps(root: nn.Module):
    name_to_module = {}
    name_to_type = {}
    for n, m in root.named_modules():
        name_to_module[n] = m
        name_to_type[n] = type(m).__name__
    return name_to_module, name_to_type

name_to_module, name_to_type = _build_name_maps(model)

def _parent_name(module_name: str) -> str:
    if not module_name:
        return ""
    if "." not in module_name:
        return ""
    return module_name.rsplit(".", 1)[0]

def _has_vision_ancestor(module_name: str) -> bool:
    current = _parent_name(module_name)
    while current:
        t = name_to_type.get(current, "").lower()
        if any(mark in t for mark in vision_type_markers):
            return True
        current = _parent_name(current)
    return False

def _should_ignore_module(module_name: str, out_features: int, in_features: int, vocab_size: int) -> bool:
    # Ignore VL vision tower and multimodal projectors via ancestor type markers
    if _has_vision_ancestor(module_name):
        return True
    # Do not quantize final output head (projection to vocab)
    if vocab_size is not None and out_features == vocab_size:
        return True
    return False

# Helper: identify linear-like modules by presence of a 2D floating-point weight
def _has_2d_weight(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if weight is None or not isinstance(weight, torch.nn.Parameter):
        return False
    try:
        return weight.ndim == 2 and weight.dtype.is_floating_point
    except Exception:
        return False

# Discover quantizable module types and names (beyond torch.nn.Linear)
included_modules = []
ignored_modules = []
type_to_included_names = defaultdict(list)
type_to_ignored_names = defaultdict(list)

# Infer vocab size for structural exclusion of lm_head
vocab_size = getattr(getattr(model, "config", object()), "vocab_size", None)
if vocab_size is None:
    try:
        vocab_size = tokenizer.vocab_size
    except Exception:
        vocab_size = None

for mod_name, mod in model.named_modules():
    if not _has_2d_weight(mod):
        continue
    type_name = type(mod).__name__
    if "linear" not in type_name.lower():
        continue
    weight = getattr(mod, "weight")
    out_features = int(weight.shape[0])
    in_features = int(weight.shape[1])

    # Structure-based ignore rules
    if _should_ignore_module(mod_name, out_features, in_features, vocab_size):
        ignored_modules.append(mod_name)
        type_to_ignored_names[type_name].append(mod_name)
        continue

    # Positive include: FFN-like expansions/compressions should be included even if names vary
    ratio = max(out_features, in_features) / max(1, min(out_features, in_features))
    is_ffn_like = ratio >= 1.5

    # Include all linear-like in transformer blocks (attention projections typically ratio ~1 or 3)
    included_modules.append(mod_name)
    type_to_included_names[type_name].append(mod_name)

awq_targets = sorted(type_to_included_names.keys()) or ["Linear"]
moe_ignores = sorted(set(ignored_modules + ["lm_head"]))

print(f"[AWQ] Detected {len(included_modules)} quantizable modules across {len(awq_targets)} types.")
print(f"[AWQ] Target types: {awq_targets[:8]}{'...' if len(awq_targets) > 8 else ''}")
top_types = Counter({t: len(names) for t, names in type_to_included_names.items()}).most_common(6)
print(f"[AWQ] Included counts by type (top): {top_types}")
print(f"[AWQ] Ignored modules: {len(moe_ignores)} (sample: {moe_ignores[:6]})")
sample_included = included_modules[:6]
print(f"[AWQ] Sample included modules: {sample_included}")

# Build AWQ type mappings: treat custom linear-like wrappers as Linear
awq_mappings = []
for t in awq_targets:
    if t != "Linear":
        awq_mappings.append({"source": t, "target": "Linear"})
print(f"[AWQ] Type mappings -> Linear: {awq_mappings[:8]}{'...' if len(awq_mappings) > 8 else ''}")

recipe = [
    AWQModifier(
        targets=["Linear"],
        ignore=moe_ignores,
        mappings=awq_mappings,
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                    "dynamic": False,
                },
            },
        },
    ),
]

# =========================
# Convert fused experts -> per-expert Linear, then Quantize + save
# =========================
SAVE_DIR = require_env("DST_DIR")

CONVERT_DIR = str(Path(SAVE_DIR) / "converted_linear")
print(f"[CONVERT] Converting fused experts to per-expert Linear into: {CONVERT_DIR}")
converted_model = _convert_fused_experts_to_linear_state_dict(model, tokenizer, CONVERT_DIR)

print("[AWQ] Starting oneshot on converted model...")
oneshot(
    model=converted_model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    calibrate_moe_context=get_env_bool("CALIBRATE_MOE_CONTEXT", False),
)

print("[SAVE] Saving compressed model...")
converted_model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

print("Saved to:", SAVE_DIR)

