# Vinyl Audio Restoration

A Python project for reducing crackle, clicks, hiss, and low-frequency rumble in digitized vinyl recordings while preserving the music.

Version 5 combines GPU-assisted noise analysis, predictive click repair, and frequency-dependent background reduction. Compare Gentle, Balanced, and Strong previews before processing the full recording.

## Hear the difference

### Original recording



https://github.com/user-attachments/assets/86cbb117-d09a-444b-8b67-d6f316a19789






### Restored recording

_Add the matching restored audio sample here.



https://github.com/user-attachments/assets/e89d3d5a-a686-4817-8d02-12c279c23a7f






## Getting started

Requires Python, NumPy, SciPy, SoundFile, and a CUDA-enabled PyTorch installation for NVIDIA GPU processing. Set `USE_GPU = False` to use the CPU.

1. Open `restore_vinyl_gpu_v5.py` in Thonny or your preferred Python editor.
2. Set `INPUT_FILE` to your recording's path.
3. Run the script and listen to the generated previews.
4. Enter `g`, `b`, or `s` in the console to choose Gentle, Balanced, or Strong.

The script saves your choice and exports a 24-bit WAV, separate removed-noise files, and a processing report. Outputs go in a **[recording name] - V5** folder beside the input. Repeated runs overwrite matching V5 outputs; the original recording is protected.

## A note on quality

Restoration is a tradeoff: stronger settings can remove musical detail along with noise. Compare previews at similar listening levels and choose the lightest processing that sounds good to you.
