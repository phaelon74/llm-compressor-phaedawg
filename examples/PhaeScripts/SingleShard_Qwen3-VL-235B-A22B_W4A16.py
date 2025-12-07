from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoProcessor
from transformers.utils import logging as hf_logging
import os
from pathlib import Path
import torch
import torch.nn as nn
import requests
import importlib.util
from accelerate import init_empty_weights
import json
import shutil
from safetensors.torch import safe_open, save_file
try:
    from auto_round import AutoRound
except Exception:
    AutoRound = None

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

print(f"Loading tokenizer/config from: {MODEL_ID}")
hf_token = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=hf_token)
print("Tokenizer loaded.")
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

def _convert_shards_offline(src_dir: str, dst_dir: str):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    index_path = src_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise RuntimeError("model.safetensors.index.json not found in SRC_DIR")
    index_data = json.loads(index_path.read_text())
    weight_map = index_data.get("weight_map", {})

    # Group param names by shard
    shard_to_keys = {}
    for name, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(name)

    # Read config for num_experts
    config = AutoConfig.from_pretrained(src_dir, trust_remote_code=True, token=hf_token)
    text_cfg = getattr(config, "text_config", None)
    num_experts = getattr(text_cfg, "num_experts", None)

    new_weight_map = {}
    total_size = 0

    for shard, keys in sorted(shard_to_keys.items()):
        src_shard_path = src_dir / shard
        if not src_shard_path.exists():
            raise RuntimeError(f"Missing shard: {src_shard_path}")
        print(f"[CONVERT] Processing shard: {shard} with {len(keys)} tensors")

        out_tensors = {}
        with safe_open(src_shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                if key not in f.keys():
                    continue
                t = f.get_tensor(key)
                # Expert fused -> per-expert
                is_expert = ("expert" in key) and ("gate_up_proj" in key or "down_proj" in key)
                if is_expert and t.ndim == 3:
                    n_experts = t.shape[0] if num_experts is None else num_experts
                    base = key.replace("gate_up_proj", "gate_up_projs") if "gate_up_proj" in key else key.replace("down_proj", "down_projs")
                    if base.endswith(".weight"):
                        base = base[: -len(".weight")]
                    for i in range(n_experts):
                        new_key = f"{base}.{i}.weight"
                        value = t[i, ...].transpose(0, 1).contiguous()
                        out_tensors[new_key] = value
                        new_weight_map[new_key] = shard
                        total_size += value.numel() * value.element_size()
                else:
                    out_tensors[key] = t
                    new_weight_map[key] = shard
                    total_size += t.numel() * t.element_size()

        # Write converted shard
        dst_shard_path = dst_dir / shard
        save_file(out_tensors, str(dst_shard_path))
        print(f"[CONVERT] Wrote shard: {dst_shard_path}")

    # Write new index
    new_index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(new_index))

    # Copy config/tokenizer aux files if present
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ]:
        src_path = src_dir / fname
        if src_path.exists():
            shutil.copy2(src_path, dst_dir / fname)

    # Download custom modeling file into converted folder for trust_remote_code
    _download_modeling_qwen3_vl_moe(dst_dir)

# =========================
# Calibration data (WikiText)
# =========================
NUM_CALIBRATION_SAMPLES = get_env_int("CALIB_SAMPLES", 256)
MAX_SEQUENCE_LENGTH = get_env_int("MAX_SEQ_LEN", 512)
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

## name_to_module/name_to_type will be built after loading the converted model

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

## AWQ targets/recipe will be built after loading the converted model

# =========================
# Offline shard-by-shard convert, then quantize (AutoRound by default) + save
# =========================
SAVE_DIR = require_env("DST_DIR")
CONVERT_DIR = os.getenv("CONVERT_DIR") or str(Path(SAVE_DIR) / "converted_linear")

convert_index = Path(CONVERT_DIR) / "model.safetensors.index.json"
if get_env_bool("SKIP_CONVERT", False) or convert_index.exists():
    print(f"[CONVERT] Skipping shard conversion; using existing: {CONVERT_DIR}")
else:
    print(f"[CONVERT] Offline shard conversion -> {CONVERT_DIR}")
    _convert_shards_offline(SRC_DIR, CONVERT_DIR)

# Ensure custom modeling file exists for trust_remote_code
if not (Path(CONVERT_DIR) / "modeling_qwen3_vl_moe.py").exists():
    _download_modeling_qwen3_vl_moe(CONVERT_DIR)

USE_AUTOROUND = get_env_bool("USE_AUTOROUND", True)

if USE_AUTOROUND:
    if AutoRound is None:
        raise RuntimeError("auto-round is not installed. pip install auto-round")
    print("[LOAD] Loading converted model for AutoRound...")
    # AutoRound expects the custom modeling class name
    modeling_path = Path(CONVERT_DIR) / "modeling_qwen3_vl_moe.py"
    if not modeling_path.exists():
        _download_modeling_qwen3_vl_moe(CONVERT_DIR)
    Qwen3VLMoeForConditionalGeneration = _load_custom_qwen3_vl_moe_class(str(modeling_path))
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(CONVERT_DIR, dtype="auto", device_map="cpu")
    processor = AutoProcessor.from_pretrained(CONVERT_DIR)
    # Build Intel-like layer_config
    layer_config = {}
    for n, m in model.named_modules():
        if type(m) == torch.nn.Linear:
            if "visual" in n or "vision" in n:
                continue
            if "mlp.gate" in n:
                layer_config[n] = {"bits": 16}
            elif "expert" in n and "shared_experts" not in n:
                layer_config[n] = {"bits": 4}
            elif n != "lm_head":
                layer_config[n] = {"bits": 8}
    print(f"[AR] Layers in policy: {len(layer_config)}")

    ar = AutoRound(model, iters=0, layer_config=layer_config, tokenizer=tokenizer, processor=processor)
    ar.quantize_and_save(format="auto_round", output_dir=SAVE_DIR)
    print("Saved to:", SAVE_DIR)
else:
    print("[LOAD] Loading converted model for AWQ...")
    model = AutoModel.from_pretrained(
        CONVERT_DIR,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Rebuild maps for converted model before AWQ
    name_to_module = { }
    name_to_type = { }
    for n, m in model.named_modules():
        name_to_module[n] = m
        name_to_type[n] = type(m).__name__

    # Discover quantizable module types and names (beyond torch.nn.Linear) AFTER load
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

        included_modules.append(mod_name)
        type_to_included_names[type_name].append(mod_name)

    awq_targets = sorted(type_to_included_names.keys()) or ["Linear"]
    moe_ignores = sorted(set(ignored_modules + ["lm_head"]))

    print(f"[AWQ] Detected {len(included_modules)} quantizable modules across {len(awq_targets)} types.")
    print(f"[AWQ] Using targets: {awq_targets[:12]}{'...' if len(awq_targets) > 12 else ''}")

    recipe = [
        AWQModifier(
            targets=awq_targets,
            ignore=moe_ignores,
            mappings=[],
            config_groups={
                "group_0": {
                    "targets": awq_targets,
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

    print("[AWQ] Starting oneshot on converted model...")
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        calibrate_moe_context=get_env_bool("CALIBRATE_MOE_CONTEXT", True),
        output_dir=SAVE_DIR,
    )
    print("Saved to:", SAVE_DIR)

