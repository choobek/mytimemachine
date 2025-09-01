"""
Unified Inference Script for MyTimeMachine
==========================================

This script consolidates all inference functionality into one clean interface:
- Automatic face alignment
- Tunable aging strength for identity preservation 
- Organized outputs with clear naming
- Multiple output formats (individual, coupled, comparison grids)

Usage:
    python scripts/inference_unified.py \
        --checkpoint_path experiments/full_training_run/00015/checkpoints/iteration_35000.pt \
        --input_dir data/inference_src \
        --output_dir inference_results \
        --target_age 39 \
        --aging_strength 0.6 \
        --create_comparisons
"""

from argparse import Namespace
import os
import re
import time
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
import dlib
import argparse

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.augmentations import AgeTransformer
from utils.common import tensor2im
from models.psp import pSp
from scripts.align_all_parallel import align_face
from configs.paths_config import model_paths

def main():
    parser = argparse.ArgumentParser(description='Unified MyTimeMachine Inference')
    
    # Required arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save results')
    parser.add_argument('--target_age', type=str, required=True,
                       help='Target age(s) for aging (comma-separated)')
    
    # Aging control
    parser.add_argument('--aging_strength', type=float, default=0.6,
                       help='Aging strength (0.0=no aging, 1.0=full aging)')
    parser.add_argument('--style_layers', type=str, default='8,9,10,11,12',
                       help='Comma-separated StyleGAN layers to modify')
    parser.add_argument('--preserve_pose_strict', action='store_true',
                       help='Stronger pose preservation (freeze coarse layers 0-7, weaken 8-9)')
    
    # Output options
    parser.add_argument('--create_comparisons', action='store_true',
                       help='Create comparison grids showing different aging strengths')
    parser.add_argument('--create_coupled', action='store_true',
                       help='Create side-by-side before/after images')
    parser.add_argument('--output_size', type=int, default=512,
                       help='Output image size (256 or 512)')
    
    # Processing options
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Batch size for inference')
    parser.add_argument('--max_images', type=int, default=None,
                       help='Maximum number of images to process')
    parser.add_argument('--skip_alignment', action='store_true',
                       help='Skip face alignment (use if images already aligned)')
    
    args = parser.parse_args()
    
    # Setup
    # Resolve versioned output directory (e.g., inference_results/v001)
    base_output_dir = args.output_dir.rstrip('/')
    versioned_output_dir, version_label = resolve_versioned_output_dir(base_output_dir)
    # From here on, use the resolved versioned directory everywhere
    args.output_dir = versioned_output_dir

    print("🚀 MyTimeMachine Unified Inference")
    print(f"📁 Input: {args.input_dir}")
    print(f"📁 Output base: {base_output_dir}")
    print(f"🏷️  Version: {version_label} → {args.output_dir}")
    print(f"🎯 Target age: {args.target_age}")
    print(f"💪 Aging strength: {args.aging_strength}")
    
    # Create organized output structure
    setup_output_directories(args)
    
    # Load model
    print("🔄 Loading model...")
    net, opts = load_model(args.checkpoint_path, args=args)
    
    # Load face alignment if needed
    predictor = None
    if not args.skip_alignment:
        print("🔄 Loading face alignment...")
        predictor = dlib.shape_predictor(model_paths["shape_predictor"])
    
    # Process images
    print("🔄 Processing images...")
    stats = process_images(net, opts, predictor, args)
    
    # Save statistics
    save_statistics(stats, args)
    
    print(f"✅ Complete! Processed {stats['successful']} images")
    print(f"📊 Results saved to: {args.output_dir}")


def resolve_versioned_output_dir(base_dir: str):
    """Create and return a new versioned output directory under base_dir.

    Example:
      base_dir='inference_results' → creates/returns 'inference_results/v001' (or next available)
    """
    os.makedirs(base_dir, exist_ok=True)
    # Find existing version subdirectories matching vNNN
    existing = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and re.match(r'^v\d{3}$', d)]
    if existing:
        nums = [int(d[1:]) for d in existing]
        next_num = max(nums) + 1
    else:
        next_num = 1
    version_label = f"v{next_num:03d}"
    full_path = os.path.join(base_dir, version_label)
    os.makedirs(full_path, exist_ok=True)
    return full_path, version_label


def setup_output_directories(args):
    """Create organized output directory structure"""
    os.makedirs(args.output_dir, exist_ok=True)
    
    for age in args.target_age.split(','):
        age = age.strip()
        
        # Main results
        os.makedirs(os.path.join(args.output_dir, f"age_{age}", "results"), exist_ok=True)
        
        # Aligned inputs (for reference)
        os.makedirs(os.path.join(args.output_dir, f"age_{age}", "aligned_inputs"), exist_ok=True)
        
        # Optional outputs
        if args.create_coupled:
            os.makedirs(os.path.join(args.output_dir, f"age_{age}", "before_after"), exist_ok=True)
        
        if args.create_comparisons:
            os.makedirs(os.path.join(args.output_dir, f"age_{age}", "comparisons"), exist_ok=True)


def load_model(checkpoint_path, args=None):
    """Load the trained model ensuring runtime args override ckpt opts.

    The ckpt stores training-time `opts`, including a `checkpoint_path` that often
    points to the base SAM model. During inference we must override this with the
    user-provided `checkpoint_path` (and any other relevant runtime args) so the
    correct personalized checkpoint is loaded.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts_dict = ckpt['opts']
    # Ensure the correct checkpoint path is used (do NOT keep the one saved in ckpt opts)
    opts_dict['checkpoint_path'] = checkpoint_path
    # Optionally merge additional runtime args that downstream code might rely on
    # Do not override architecture-critical fields (e.g., output_size).
    # Keep only the explicit checkpoint_path override.
    opts = Namespace(**opts_dict)

    net = pSp(opts)
    net.eval()
    net.cuda()

    return net, opts


def process_images(net, opts, predictor, args):
    """Process all images with unified pipeline"""
    
    # Get input images
    input_images = get_input_images(args.input_dir)
    if args.max_images:
        input_images = input_images[:args.max_images]
    
    print(f"Found {len(input_images)} images to process")
    
    # Parse style layers
    style_layers = [int(x.strip()) for x in args.style_layers.split(',')]
    
    # Initialize statistics
    stats = {
        'total': len(input_images),
        'successful': 0,
        'failed_alignment': 0,
        'failed_inference': 0,
        'processing_times': []
    }
    
    # Create age transformers
    age_transformers = [AgeTransformer(target_age=age.strip()) 
                       for age in args.target_age.split(',')]
    
    # Process each age
    for age_transformer in age_transformers:
        age_str = str(age_transformer.target_age)
        print(f"\n🎯 Processing target age: {age_str}")
        
        for img_name in tqdm(input_images, desc=f"Age {age_str}"):
            try:
                # Step 1: Load and align image
                aligned_image, success = load_and_align_image(
                    os.path.join(args.input_dir, img_name), 
                    predictor, 
                    args.skip_alignment
                )
                
                if not success:
                    stats['failed_alignment'] += 1
                    continue
                
                # Step 2: Save aligned input for reference
                aligned_path = os.path.join(args.output_dir, f"age_{age_str}", "aligned_inputs", img_name)
                aligned_image.save(aligned_path)
                
                # Step 3: Run inference
                start_time = time.time()
                
                result_image = run_unified_inference(
                    aligned_image, age_transformer, net, opts, 
                    args.aging_strength, style_layers, args.output_size, args.preserve_pose_strict
                )
                
                processing_time = time.time() - start_time
                stats['processing_times'].append(processing_time)
                
                # Step 4: Save results
                save_results(result_image, aligned_image, img_name, age_str, args)
                
                stats['successful'] += 1
                
            except Exception as e:
                print(f"❌ Failed to process {img_name}: {str(e)}")
                stats['failed_inference'] += 1
                continue
    
    return stats


def get_input_images(input_dir):
    """Get list of image files from input directory"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    images = []
    
    for ext in image_extensions:
        images.extend([f for f in os.listdir(input_dir) 
                      if f.lower().endswith(ext)])
    
    return sorted(images)


def load_and_align_image(image_path, predictor, skip_alignment):
    """Load image and optionally align face"""
    try:
        if skip_alignment:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((256, 256))
            return image, True
        else:
            aligned_image = align_face(filepath=image_path, predictor=predictor)
            aligned_image = aligned_image.resize((256, 256))
            return aligned_image, True
    except Exception as e:
        print(f"❌ Alignment failed for {os.path.basename(image_path)}: {e}")
        return None, False


def run_unified_inference(aligned_image, age_transformer, net, opts, aging_strength, style_layers, output_size, preserve_pose_strict=False):
    """Run inference with tunable aging strength"""
    
    # Prepare input tensor
    transform = data_configs.DATASETS['ffhq_aging']['transforms'](opts).get_transforms()['transform_inference']
    input_tensor = transform(aligned_image).unsqueeze(0)
    
    with torch.no_grad():
        # Apply age transformation
        input_age_batch = age_transformer(input_tensor.squeeze(0).cpu()).unsqueeze(0).to('cuda')
        input_cuda = input_age_batch.cuda().float()
        
        # Create identity-preserving input
        original_rgb = input_cuda[:, :3, :, :]
        current_age_channel = input_cuda[:, -1:, :, :]
        original_age = torch.zeros_like(current_age_channel)
        
        # Blend ages based on aging strength
        blended_age = (1 - aging_strength) * original_age + aging_strength * current_age_channel
        identity_input = torch.cat([original_rgb, blended_age], dim=1)
        
        # Get latents
        _, aged_latents = net(input_cuda, return_latents=True, randomize_noise=False)
        _, identity_latents = net(identity_input, return_latents=True, randomize_noise=False)
        
        # Start from identity latents to better preserve pose/eye direction
        hybrid_latents = identity_latents.clone()
        n_styles = hybrid_latents.shape[1]
        selected = set(style_layers)
        for i in range(n_styles):
            if preserve_pose_strict:
                # Freeze coarse (0-7), weaken transition (8-9), allow edits mid (10-12), limit fine (>=13)
                if i <= 7:
                    alpha = 0.0
                elif 8 <= i <= 9:
                    alpha = aging_strength * 0.3
                elif 10 <= i <= 12:
                    alpha = aging_strength
                else:
                    alpha = min(aging_strength, 0.4)
            else:
                alpha = aging_strength if i in selected else 0.0
            if alpha > 0:
                hybrid_latents[:, i] = (1 - alpha) * identity_latents[:, i] + alpha * aged_latents[:, i]
        
        # Generate final image
        result_batch, _ = net.decoder([hybrid_latents], input_is_latent=True, randomize_noise=False)
        
        # Resize to desired output size
        if output_size == 256:
            result_batch = net.face_pool(result_batch)
        
        return tensor2im(result_batch[0])


def save_results(result_image, aligned_image, img_name, age_str, args):
    """Save all requested output formats"""
    
    # Main result
    result_path = os.path.join(args.output_dir, f"age_{age_str}", "results", img_name)
    result_image.save(result_path)
    
    # Before/after comparison
    if args.create_coupled:
        before_after = create_before_after(aligned_image, result_image, args.output_size)
        coupled_path = os.path.join(args.output_dir, f"age_{age_str}", "before_after", img_name)
        before_after.save(coupled_path)
    
    # Comparison grid (multiple aging strengths)
    if args.create_comparisons:
        comparison_grid = create_comparison_grid(aligned_image, result_image, args.output_size)
        comparison_path = os.path.join(args.output_dir, f"age_{age_str}", "comparisons", img_name)
        comparison_grid.save(comparison_path)


def create_before_after(aligned_image, result_image, output_size):
    """Create side-by-side before/after comparison"""
    aligned_resized = aligned_image.resize((output_size, output_size))
    result_resized = result_image.resize((output_size, output_size))
    
    before_after = Image.new('RGB', (output_size * 2, output_size))
    before_after.paste(aligned_resized, (0, 0))
    before_after.paste(result_resized, (output_size, 0))
    
    return before_after


def create_comparison_grid(aligned_image, result_image, output_size):
    """Create comparison showing original + result (placeholder for future expansion)"""
    # For now, just return the before/after
    # This can be expanded to show multiple aging strengths
    return create_before_after(aligned_image, result_image, output_size)


def save_statistics(stats, args):
    """Save processing statistics"""
    stats_path = os.path.join(args.output_dir, 'processing_stats.txt')
    
    avg_time = np.mean(stats['processing_times']) if stats['processing_times'] else 0
    std_time = np.std(stats['processing_times']) if stats['processing_times'] else 0
    
    stats_content = f"""MyTimeMachine Inference Statistics
=====================================

Input Directory: {args.input_dir}
Output Directory: {args.output_dir}
Target Ages: {args.target_age}
Aging Strength: {args.aging_strength}
Style Layers: {args.style_layers}

Processing Results:
- Total images: {stats['total']}
- Successfully processed: {stats['successful']}
- Failed alignment: {stats['failed_alignment']}
- Failed inference: {stats['failed_inference']}
- Success rate: {stats['successful']/stats['total']*100:.1f}%

Performance:
- Average processing time: {avg_time:.3f}s ± {std_time:.3f}s
- Total processing time: {sum(stats['processing_times']):.1f}s

Output Structure:
{args.output_dir}/
├── age_{{age}}/
│   ├── results/          # Main aged results
│   ├── aligned_inputs/   # Aligned input images
{'│   ├── before_after/    # Side-by-side comparisons' if args.create_coupled else ''}
{'│   └── comparisons/     # Comparison grids' if args.create_comparisons else ''}
└── processing_stats.txt  # This file
"""
    
    with open(stats_path, 'w') as f:
        f.write(stats_content)
    
    print(f"📊 Statistics saved to: {stats_path}")


if __name__ == '__main__':
    main()
