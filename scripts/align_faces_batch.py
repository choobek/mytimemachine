import os
import dlib
from tqdm import tqdm
from PIL import Image
import sys
sys.path.append(".")
sys.path.append("..")

from scripts.align_all_parallel import align_face
from configs.paths_config import model_paths

def align_all_faces(input_dir, output_dir):
    """
    Align all faces in input directory and save successfully aligned ones to output directory
    Returns list of successfully processed images
    """
    # Load dlib shape predictor for face alignment
    predictor = dlib.shape_predictor(model_paths["shape_predictor"])
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of input images
    input_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        input_images.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    input_images.sort()
    
    successful_images = []
    failed_images = []
    
    print(f"Processing {len(input_images)} images for face alignment...")
    
    for img_name in tqdm(input_images):
        img_path = os.path.join(input_dir, img_name)
        
        try:
            # Align the face
            aligned_image = align_face(filepath=img_path, predictor=predictor)
            aligned_image = aligned_image.resize((256, 256))
            
            # Save aligned image
            aligned_save_path = os.path.join(output_dir, img_name)
            aligned_image.save(aligned_save_path)
            
            successful_images.append(img_name)
            
        except Exception as e:
            print(f"Failed to align {img_name}: {str(e)}")
            failed_images.append(img_name)
            continue
    
    # Save statistics
    stats_path = os.path.join(output_dir, 'alignment_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f"Total images: {len(input_images)}\n")
        f.write(f"Successfully aligned: {len(successful_images)}\n")
        f.write(f"Failed to align: {len(failed_images)}\n")
        f.write(f"Success rate: {len(successful_images)/len(input_images)*100:.1f}%\n\n")
        f.write("Successfully aligned images:\n")
        for img in successful_images:
            f.write(f"  {img}\n")
        f.write("\nFailed images:\n")
        for img in failed_images:
            f.write(f"  {img}\n")
    
    print(f"\nAlignment complete!")
    print(f"Successfully aligned: {len(successful_images)}/{len(input_images)} ({len(successful_images)/len(input_images)*100:.1f}%)")
    print(f"Results saved to: {output_dir}")
    print(f"Statistics saved to: {stats_path}")
    
    return successful_images, failed_images

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save aligned images')
    args = parser.parse_args()
    
    align_all_faces(args.input_dir, args.output_dir)
