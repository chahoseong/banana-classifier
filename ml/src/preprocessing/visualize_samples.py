import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier")
from ml.src.preprocessing import image_processor

def generate_visualizations():
    base_raw = r"d:\Intel_AI_App_Creator_Hands_on\banana_classfier\banana-classifier\dataset\raw"
    artifact_dir = r"C:\Users\User\.gemini\antigravity\brain\0c7712be-d554-47b0-8462-65753e74891f"
    
    samples = [
        {"cat": "Unripe", "file": "20230204_083009851.jpg", "target": "unripe_comparison.jpg"},
        {"cat": "Ripe", "file": "20230204_083026344.jpg", "target": "ripe_comparison.jpg"},
        {"cat": "Overripe", "file": "20230204_082949.jpg", "target": "overripe_comparison.jpg"}
    ]
    
    for sample in samples:
        img_path = os.path.join(base_raw, sample["cat"], sample["file"])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load {img_path}")
            continue
            
        # Resize for consistency
        img = cv2.resize(img, (300, 300))
        
        # Apply masking (Hybrid Contour + ROI GrabCut)
        mask = image_processor.create_banana_mask(img)
        masked_img = cv2.bitwise_and(img, img, mask=mask)
        
        # Create comparison (Original | Masked)
        # Add white divider
        divider = np.ones((300, 10, 3), dtype=np.uint8) * 255
        comparison = np.hstack((img, divider, masked_img))
        
        # Save to artifacts
        save_path = os.path.join(artifact_dir, sample["target"])
        cv2.imwrite(save_path, comparison)
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    generate_visualizations()
