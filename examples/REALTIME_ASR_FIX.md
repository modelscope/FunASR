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

### Related Issue
- Closes #2720
