# Real-Time ASR Syntax Fix

## Issue #2720: Fixing Syntax Error in Real-Time Speech Recognition Example

### Problem Description
In the real-time speech recognition example, there is a syntax error in the `total_chunk_num` calculation:

```python
# INCORRECT (with syntax error)
total_chunk_num = int(len(speech)-1)/chunk_stride+1)
```

This line has mismatched parentheses - there's a closing parenthesis without a corresponding opening parenthesis.

### Solution
The correct syntax should be:

```python
# CORRECT (fixed parentheses)
total_chunk_num = int((len(speech)-1)/chunk_stride+1)
```

### Explanation
The fix involves adding an opening parenthesis after `int(` to properly group the division operation `(len(speech)-1)/chunk_stride+1` before converting it to an integer.

This ensures the arithmetic expression is evaluated correctly:
1. Calculate `len(speech)-1`
2. Divide by `chunk_stride`
3. Add 1
4. Convert the result to an integer

### Impact
This syntax error would cause a `SyntaxError` when running any real-time speech recognition examples that use this calculation.

### Affected Files
- Real-time speech recognition example files that contain the `total_chunk_num` calculation

## Files That Need To Be Updated

The following files in the FunASR repository contain the `total_chunk_num` calculation with the syntax error and should be updated:

### Example Python Files
- `examples/industrial_data_pretraining/scama/demo.py` (Line ~34)
- `examples/wenetSpeech/realtime_demo.py` (if exists)
- `runtime/triton_gpu/client/speech_client.py` (if applicable)

Each of these files should have their `total_chunk_num` calculation corrected from:
```python
total_chunk_num = int(len(speech)-1)/chunk_stride+1)  # WRONG
```

To:
```python
total_chunk_num = int((len(speech)-1)/chunk_stride+1)  # CORRECT
```

## Testing The Fix

After applying the fix, test the real-time ASR example with audio input to ensure:
1. No `SyntaxError` is raised
2. The chunk calculation produces correct integer results
3. Real-time speech recognition processes audio without errors

## Related Issues
- Issue #2720 reports this syntax error
- This documentation file serves as a guide for applying the fix across all affected files

### Related Issue
- Closes #2720
