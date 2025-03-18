import os
import cv2
import numpy as np
from glob import glob

def create_denoising_video(source_dir='/home/yujiro/venv/diffusion_model/data_betaLinear0.05_100steps', output_path='/home/yujiro/venv/diffusion_model/data_betaLinear0.05_100steps/denoising_process.mp4', fps=8):
    """
    Create a video from a series of denoising process images.
    
    Args:
        source_dir: Directory containing the denoising images
        output_path: Path where the output video will be saved
        fps: Frames per second for the output video
    """
    # Get all denoising images
    image_files = glob(os.path.join(source_dir, "denoising_*.jpg"))
    
    # Sort the image files by step number
    image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not image_files:
        print(f"No denoising images found in {source_dir}")
        return
    
    # Read the first image to get dimensions
    first_img = cv2.imread(image_files[0])
    height, width, layers = first_img.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add each image to the video
    for img_path in image_files:
        print(f"Adding frame: {img_path}")
        img = cv2.imread(img_path)
        if img is not None:
            video.write(img)
        else:
            print(f"Warning: Could not read {img_path}")
    
    # Release the video writer
    video.release()
    print(f"Video created at {output_path}")

if __name__ == "__main__":
    create_denoising_video()