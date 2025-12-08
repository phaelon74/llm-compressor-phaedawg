# Installing LLM Compressor from Source

This guide provides detailed instructions for installing LLM Compressor from source, which is required to use the FlexQ 6-bit quantization feature and other latest developments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Testing FlexQ](#testing-flexq)
5. [Troubleshooting](#troubleshooting)
6. [Development Mode](#development-mode)

## Prerequisites

### System Requirements

- **Python**: 3.10, 3.11, or 3.12 (Python 3.8-3.11 are officially supported)
- **Operating System**: Linux (recommended), macOS, or Windows with WSL2
- **CUDA**: If using GPU acceleration, CUDA 11.8 or later (for NVIDIA GPUs)
- **Git**: For cloning the repository
- **pip**: Python package installer (usually comes with Python)

### Check Your Python Version

```bash
python3 --version
# Should show Python 3.10.x, 3.11.x, or 3.12.x
```

If you don't have Python 3.10+, install it using:
- **Linux**: `sudo apt-get install python3.10` (or use pyenv)
- **macOS**: `brew install python@3.10` or download from python.org
- **Windows**: Download from python.org

### Check CUDA (Optional, for GPU support)

```bash
nvidia-smi
# Should show your GPU and CUDA version
```

If you don't have CUDA installed, you can still use CPU-only mode, but GPU acceleration is recommended for quantization.

## Installation Steps

### Step 1: Clone the Repository

Clone the LLM Compressor repository from GitHub:

```bash
# Clone the main repository
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor
```

**Note**: If you're working with a fork or specific branch with FlexQ support:

```bash
# For a specific branch
git clone -b <branch-name> https://github.com/vllm-project/llm-compressor.git
cd llm-compressor

# Or if you have a fork
git clone https://github.com/<your-username>/llm-compressor.git
cd llm-compressor
```

### Step 2: Set Up a Virtual Environment (Recommended)

Using a virtual environment isolates your LLM Compressor installation and prevents conflicts with other Python packages.

#### Option A: Using venv (Built-in, Recommended)

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

#### Option B: Using conda

```bash
# Create a conda environment
conda create -n llmcompressor python=3.10
conda activate llmcompressor
```

#### Option C: Using virtualenv

```bash
# Install virtualenv if not already installed
pip install virtualenv

# Create virtual environment
virtualenv venv

# Activate (same as venv)
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

**Verify activation**: Your terminal prompt should show `(venv)` or `(llmcompressor)`.

### Step 3: Upgrade pip and Install Build Tools

```bash
# Upgrade pip to the latest version
python -m pip install --upgrade pip

# Install build tools (required for some dependencies)
pip install wheel setuptools
```

### Step 4: Install LLM Compressor in Editable Mode

Install LLM Compressor in editable/development mode so changes to the source code are immediately available:

```bash
# Install LLM Compressor with all dependencies
pip install -e ".[dev]"
```

**What this does:**
- `-e`: Editable/development mode - changes to source code are immediately available
- `.[dev]`: Installs the package with development dependencies (testing, linting, etc.)

**If you only need runtime dependencies** (without dev tools):

```bash
pip install -e .
```

**Installation time**: This may take 5-15 minutes depending on your internet connection and system, as it downloads and compiles various dependencies including PyTorch.

### Step 5: Verify Installation

Check that LLM Compressor is installed correctly:

```bash
# Check if llmcompressor is installed
python -c "import llmcompressor; print(llmcompressor.__version__)"

# Check if FlexQ modifier is available
python -c "from llmcompressor.modifiers.flexq import FlexQModifier; print('FlexQ installed successfully!')"
```

You should see the version number and a success message.

## Verification

### Quick Test: Import Check

```bash
python -c "
from llmcompressor import oneshot
from llmcompressor.modifiers.flexq import FlexQModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs
print('✓ All imports successful!')
print('✓ FlexQ is available!')
"
```

### Check Installed Packages

```bash
pip list | grep llmcompressor
pip list | grep compressed-tensors
pip list | grep torch
```

You should see:
- `llmcompressor` (with version, likely ending in `.dev...` for source installs)
- `compressed-tensors` (dependency)
- `torch` (PyTorch)

## Testing FlexQ

### Run the Example

Test FlexQ with the provided example:

```bash
# Navigate to the examples directory
cd examples/flexq

# Run the example (this will download a model and quantize it)
python llama_example.py
```

**Note**: The first run will download the model and calibration dataset, which may take some time and require several GB of disk space.

### Minimal Test Script

Create a simple test to verify FlexQ works:

```python
# test_flexq.py
from llmcompressor.modifiers.flexq import FlexQModifier
from compressed_tensors.quantization import QuantizationScheme, QuantizationArgs

# Create a FlexQ modifier instance
modifier = FlexQModifier(
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
        ),
    },
    w_group_size=128,
)

print("✓ FlexQ modifier created successfully!")
print(f"✓ Weight group size: {modifier.w_group_size}")
print(f"✓ Activation group size: {modifier.a_group_size}")
print(f"✓ Selective activation: {modifier.enable_selective_activation}")
```

Run it:

```bash
python test_flexq.py
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "No module named 'llmcompressor'"

**Solution**: Make sure you've activated your virtual environment and installed the package:

```bash
# Verify virtual environment is activated
which python  # Should show path to venv/bin/python

# Reinstall
pip install -e ".[dev]"
```

#### Issue 2: "ERROR: Could not build wheels for..."

**Solution**: Install build dependencies:

```bash
# On Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# On macOS
xcode-select --install

# Then retry installation
pip install -e ".[dev]"
```

#### Issue 3: CUDA/PyTorch version conflicts

**Solution**: Install PyTorch separately first, then install LLM Compressor:

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install LLM Compressor
pip install -e ".[dev]"
```

#### Issue 4: "Permission denied" errors

**Solution**: Don't use `sudo` with pip in a virtual environment. If you must, use `--user` flag:

```bash
pip install --user -e ".[dev]"
```

#### Issue 5: Import errors for FlexQ

**Solution**: Verify the FlexQ files exist:

```bash
ls src/llmcompressor/modifiers/flexq/
# Should show: __init__.py, base.py, fine_grained_group.py, layer_sensitivity.py, mappings.py
```

If files are missing, make sure you cloned the correct branch or have the latest changes.

#### Issue 6: "compressed-tensors version mismatch"

**Solution**: Update compressed-tensors:

```bash
pip install --upgrade "compressed-tensors>=0.12.3a2"
```

#### Issue 7: Out of memory during installation

**Solution**: Install without dev dependencies first, then add them:

```bash
# Install minimal dependencies
pip install -e .

# Then add dev dependencies if needed
pip install -e ".[dev]"
```

### Getting Help

If you encounter issues not covered here:

1. **Check the Issues**: Search [GitHub Issues](https://github.com/vllm-project/llm-compressor/issues) for similar problems
2. **Check Documentation**: Review [DEVELOPING.md](DEVELOPING.md) and [CONTRIBUTING.md](CONTRIBUTING.md)
3. **Ask for Help**: Open a new issue on GitHub with:
   - Your Python version (`python --version`)
   - Your OS and version
   - The full error message
   - Steps to reproduce

## Development Mode

If you plan to modify LLM Compressor source code:

### Install with Development Dependencies

```bash
pip install -e ".[dev]"
```

This includes:
- Testing frameworks (pytest)
- Linting tools (ruff, mypy)
- Documentation tools (mkdocs)
- Pre-commit hooks

### Run Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/path/to/test_file.py

# Run with verbose output
pytest -v
```

### Code Quality Checks

```bash
# Format code
make style

# Check code quality
make quality
```

### Update Installation After Code Changes

Since you installed in editable mode (`-e`), code changes are immediately available. However, if you add new dependencies:

```bash
# Reinstall to pick up new dependencies
pip install -e ".[dev]"
```

## Uninstallation

To remove LLM Compressor:

```bash
# Deactivate virtual environment first
deactivate

# Remove the package
pip uninstall llmcompressor

# Optionally remove the virtual environment
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

## Next Steps

After successful installation:

1. **Try the FlexQ Example**: Run `examples/flexq/llama_example.py`
2. **Read the Documentation**: Check `examples/flexq/README.md` for usage details
3. **Explore Other Features**: Browse other examples in the `examples/` directory
4. **Read the Paper**: [FlexQ: Efficient Post-training INT6 Quantization](https://arxiv.org/abs/2508.04405)

## Summary

Quick installation command sequence:

```bash
# 1. Clone repository
git clone https://github.com/vllm-project/llm-compressor.git
cd llm-compressor

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR: venv\Scripts\activate  # Windows

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install LLM Compressor
pip install -e ".[dev]"

# 5. Verify installation
python -c "from llmcompressor.modifiers.flexq import FlexQModifier; print('Success!')"
```

That's it! You're ready to use FlexQ 6-bit quantization with LLM Compressor.

