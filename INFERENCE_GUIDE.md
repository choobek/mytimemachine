# MyTimeMachine Inference Guide

## 🚀 Quick Start

Use the unified inference script for all your aging needs:

```bash
python scripts/inference_unified.py \
    --checkpoint_path experiments/full_training_run/00015/checkpoints/iteration_35000.pt \
    --input_dir data/your_images \
    --output_dir results \
    --target_age 39 \
    --aging_strength 0.5 \
    --create_coupled \
    --output_size 512
```

## 📁 Clean Output Structure

The unified script creates a well-organized output structure:

```
results/
├── age_39/
│   ├── results/          # Main aged results ⭐
│   ├── aligned_inputs/   # Face-aligned input images
│   ├── before_after/     # Side-by-side comparisons
│   └── comparisons/      # Comparison grids
└── processing_stats.txt  # Processing statistics
```

## ⚙️ Key Parameters

### Identity Preservation
- `--aging_strength 0.3`: Conservative aging, strong identity preservation
- `--aging_strength 0.5`: Balanced aging (recommended)
- `--aging_strength 0.8`: Strong aging, less identity preservation

### Output Options
- `--create_coupled`: Generate before/after side-by-side images
- `--create_comparisons`: Generate comparison grids
- `--output_size 512`: High quality output (or 256 for faster processing)

### Face Alignment
- Default: Automatic face alignment using dlib
- `--skip_alignment`: Skip alignment if images are already aligned

## 📋 Available Scripts

### Primary Scripts
- **`inference_unified.py`** ⭐ - Main inference script with all features
- `inference.py` - Original basic inference (for reference)

### Utility Scripts  
- `align_faces_batch.py` - Batch face alignment utility
- `align_all_parallel.py` - Core face alignment functions

### Specialized Scripts
- `inference_side_by_side.py` - Side-by-side comparisons only
- `reference_guided_inference.py` - Reference-guided aging

## 🎯 Examples

### Basic Aging
```bash
python scripts/inference_unified.py \
    --checkpoint_path experiments/full_training_run/00015/checkpoints/iteration_35000.pt \
    --input_dir data/inference_src \
    --output_dir my_results \
    --target_age 39
```

### High-Quality with Identity Preservation
```bash
python scripts/inference_unified.py \
    --checkpoint_path experiments/full_training_run/00015/checkpoints/iteration_35000.pt \
    --input_dir data/inference_src \
    --output_dir my_results \
    --target_age 39 \
    --aging_strength 0.4 \
    --create_coupled \
    --create_comparisons \
    --output_size 512
```

### Multiple Ages
```bash
python scripts/inference_unified.py \
    --checkpoint_path experiments/full_training_run/00015/checkpoints/iteration_35000.pt \
    --input_dir data/inference_src \
    --output_dir my_results \
    --target_age "30,39,45,55" \
    --aging_strength 0.5 \
    --create_coupled
```

### Pre-aligned Images (Faster)
```bash
python scripts/inference_unified.py \
    --checkpoint_path experiments/full_training_run/00015/checkpoints/iteration_35000.pt \
    --input_dir data/inference_aligned \
    --output_dir my_results \
    --target_age 39 \
    --skip_alignment \
    --aging_strength 0.5
```

## 🔧 Troubleshooting

### Face Alignment Issues
If many images fail alignment:
1. Use `scripts/align_faces_batch.py` first to check alignment success rate
2. Consider using different images with clearer face views
3. For pre-aligned images, use `--skip_alignment`

### Identity Preservation Issues
If the aged person doesn't look similar enough:
1. Reduce `--aging_strength` (try 0.3-0.4)
2. Modify `--style_layers` to focus on specific features
3. Check if your training used sufficient identity loss

### Performance Optimization
- Use `--batch_size 4` for faster processing (if GPU memory allows)
- Use `--output_size 256` for faster processing
- Use `--max_images N` to process only N images for testing

## 📊 Understanding Results

- **`results/`**: The main aged images you want
- **`aligned_inputs/`**: Check if face alignment worked correctly  
- **`before_after/`**: Easy visual comparison of results
- **`processing_stats.txt`**: Success rates and performance metrics

## 🧹 Archive

Previous experimental results are preserved in:
- `inference_results_archive/` - Previous comprehensive results with multiple aging strengths
