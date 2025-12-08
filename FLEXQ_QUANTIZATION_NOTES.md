# FlexQ Quantization Notes

## Key Findings

### 1. FlexQ Does NOT Use AWQ-Style Smoothing
- **Correct**: FlexQ uses fine-grained group quantization directly, not AWQ-style smoothing
- FlexQ's approach:
  - Fine-grained group quantization (group_size=128) for weights
  - Selective 8-bit activations for sensitive layers (W6A8)
  - Custom bit-level packing for 6-bit weights
  - Specialized CUDA kernels for W6Ax inference

### 2. Model Size Issue (8.6GB vs Expected ~6GB)

**Expected Size for 8B Model:**
- FP16: 8B params × 2 bytes = 16GB
- W6 (theoretical): 8B params × 6/8 bytes = 6GB
- **Current**: 8.6GB

**Possible Causes:**

1. **compressed-tensors may not fully support 6-bit packing**
   - The library might be storing weights in INT8 format instead of INT6
   - Or storing weights in FP16 with quantization metadata
   - FlexQ uses custom bit-level packing that may not be implemented in compressed-tensors

2. **Weights might not be quantized before saving**
   - Quantization status is FROZEN (not COMPRESSED)
   - `save_compressed=True` should handle compression, but may not work correctly for 6-bit

3. **Storage format overhead**
   - Scales and zero points add overhead
   - Group quantization metadata increases size
   - But this shouldn't account for 2.6GB difference

### 3. FlexQ's Actual Implementation

Based on the FlexQ paper (https://arxiv.org/html/2508.04405v2):

1. **Quantization Process:**
   - Uniform 6-bit weight quantization with fine-grained groups (128 elements)
   - Adaptive 8-bit activation quantization for sensitive layers
   - Calibration-free approach (no external dataset needed)
   - Bit-level packing for efficient storage

2. **Storage Format:**
   - Weights are quantized offline and packed using bit-level packing
   - Custom format optimized for 6-bit weights
   - May require custom serialization/deserialization

3. **Inference:**
   - Uses Binary Tensor Core (BTC) equivalents for INT6 operations
   - Custom CUDA kernels for W6A6 and W6A8
   - Fused dequantization in GEMM kernels

### 4. What's Missing?

The current implementation:
- ✅ Uses fine-grained group quantization (group_size=128)
- ✅ Configures 6-bit weights and 8-bit activations
- ✅ Calibrates weights during forward passes
- ❌ May not properly pack weights to 6-bit format
- ❌ Doesn't use FlexQ's custom bit-level packing
- ❌ Doesn't include FlexQ's custom CUDA kernels

### 5. Recommendations

1. **Verify Weight Format:**
   ```python
   # After quantization, check actual weight dtype and values
   for name, module in quantized_modules:
       if hasattr(module, 'weight'):
           print(f"{name}: weight dtype = {module.weight.dtype}")
           print(f"{name}: weight min/max = {module.weight.min()}, {module.weight.max()}")
   ```

2. **Check compressed-tensors Support:**
   - Verify if compressed-tensors properly supports 6-bit quantization
   - May need to use INT8 format and manually pack to 6-bit
   - Or implement custom packing similar to FlexQ

3. **FlexQ's Example:**
   - The FlexQ repository doesn't appear to have public quantization examples
   - Their implementation likely uses custom packing and kernels
   - May require implementing custom serialization for 6-bit weights

4. **Alternative Approach:**
   - Use INT8 quantization (which compressed-tensors fully supports)
   - Or implement custom 6-bit packing based on FlexQ's paper
   - Or wait for compressed-tensors to add full 6-bit support

## Next Steps

1. Verify if weights are actually quantized to 6-bit integers
2. Check if compressed-tensors has 6-bit packing support
3. Consider implementing custom packing if needed
4. Test with a smaller model first to verify quantization works correctly

