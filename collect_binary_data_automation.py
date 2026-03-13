import os
import cv2
import uuid
import shutil
import numpy as np
from pathlib import Path
from icrawler.builtin import BingImageCrawler
import logging

def configure_crawler_logger():
    # Set icrawler logging level to WARNING to avoid cluttered output
    logger = logging.getLogger('icrawler')
    logger.setLevel(logging.WARNING)

# 1. Directories setup
raw_dir = Path('dataset/raw')
binary_check_dir = Path('dataset/binary_check')
banana_dir = binary_check_dir / 'banana'
not_banana_dir = binary_check_dir / 'not_banana'

# Ensure directories exist
banana_dir.mkdir(parents=True, exist_ok=True)
not_banana_dir.mkdir(parents=True, exist_ok=True)

# Helper function: Resize, Pad to 300x300 BGR, and save with UUID
def process_and_save(img_path, output_dir, prefix=""):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False
            
        h, w = img.shape[:2]
        size = (300, 300)
        scale = min(size[0]/w, size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        # Use INTER_AREA for shrinking, INTER_CUBIC for enlarging
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        
        pad_top = (size[1] - new_h) // 2
        pad_bot = size[1] - new_h - pad_top
        pad_left = (size[0] - new_w) // 2
        pad_right = size[0] - new_w - pad_left
        
        padded_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # Extension fallback
        ext = img_path.suffix if img_path.suffix else '.jpg'
        # Unique ID to prevent overwriting
        unique_id = str(uuid.uuid4())[:8]
        new_filename = f"{prefix}_{unique_id}{ext}"
        
        cv2.imwrite(str(output_dir / new_filename), padded_img)
        return True
    except Exception as e:
        return False

print("========================================")
print("1. Checking & Copying Raw Data Integrity")
print("========================================")
# Check raw integrity before
raw_files_before = list(raw_dir.glob('**/*.*')) if raw_dir.exists() else []
raw_count_before = len(raw_files_before)
print(f"[*] Found {raw_count_before} images in dataset/raw/ before processing.")

# Copy and process raw images unconditionally (Safe Copy)
propagated_raw_count = 0
if raw_dir.exists():
    for class_folder in raw_dir.iterdir():
        if class_folder.is_dir(): 
            for img_path in class_folder.glob('*.*'):
                if process_and_save(img_path, banana_dir, prefix=f"raw_{class_folder.name}"):
                    propagated_raw_count += 1
print(f"[+] Safe copied {propagated_raw_count} images from raw to binary_check/banana/.")

# Check raw integrity after (Ensure 100% untouched)
raw_files_after = list(raw_dir.glob('**/*.*')) if raw_dir.exists() else []
raw_count_after = len(raw_files_after)
raw_integrity = "PASSED" if raw_count_before == raw_count_after else "FAILED"
if raw_integrity == "FAILED":
    print("[!] WARNING: Raw dataset integrity compromised!")

# Helper function to crawl and process
def crawl_and_process(queries, images_per_query, output_dir, prefix):
    configure_crawler_logger()
    count = 0
    for idx, q in enumerate(queries, 1):
        print(f"  [{idx}/{len(queries)}] Crawling: '{q}'...")
        temp_dir = Path(f'temp_crawl_{uuid.uuid4().hex[:6]}')
        temp_dir.mkdir(exist_ok=True)
        
        crawler = BingImageCrawler(storage={'root_dir': str(temp_dir)}, downloader_threads=4)
        try:
            crawler.crawl(keyword=q, filters=None, offset=0, max_num=images_per_query)
        except Exception as e:
            print(f"    [-] Error crawling {q}: {e}")
            
        # Process downloaded images
        for img_path in temp_dir.glob('*.*'):
            if process_and_save(img_path, output_dir, prefix=prefix):
                count += 1
            
        # Clean up temp directory immediately
        shutil.rmtree(temp_dir, ignore_errors=True)
    return count

print("\n========================================")
print("2. Crawling Generic Bananas")
print("========================================")
banana_queries = [
    "fresh yellow banana bunch", 
    "supermarket banana", 
    "perfect yellow banana", 
    "bananas on a table", 
    "single banana"
]
# 5 queries x 100 = 500 images
dl_banana_count = crawl_and_process(banana_queries, 100, banana_dir, "crawled_banana")

print("\n========================================")
print("3. Crawling 'Not Banana' (Yellow Objects)")
print("========================================")
yellow_queries = [
    "yellow post-it", "yellow toy car", "yellow highlighter", 
    "yellow mug", "yellow lemon", "yellow ball", 
    "yellow rubber duck", "yellow notebook"
]
# 8 queries x 100 = 800 images
dl_yellow_count = crawl_and_process(yellow_queries, 100, not_banana_dir, "notbanana_yellow")

print("\n========================================")
print("4. Crawling 'Not Banana' (Backgrounds)")
print("========================================")
bg_queries = [
    "desk background", "keyboard", "laptop", "human hands", 
    "empty room", "kitchen counter", "indoor room"
]
# 7 queries x 100 = 700 images
dl_bg_count = crawl_and_process(bg_queries, 100, not_banana_dir, "notbanana_bg")

# Calculate Final Stats
final_banana = len(list(banana_dir.glob('*.*')))
final_not_banana = len(list(not_banana_dir.glob('*.*')))

# We want ratio based on successful collections
total_not_banana_crawled = dl_yellow_count + dl_bg_count
yellow_ratio = (dl_yellow_count / total_not_banana_crawled * 100) if total_not_banana_crawled > 0 else 0

print("\n\n########################################")
print("           FINAL SUMMARY REPORT         ")
print("########################################")
print(f"[1] Raw Data Integrity  : {raw_integrity}")
print(f"    - Files Before      : {raw_count_before}")
print(f"    - Files After       : {raw_count_after}")
print("-" * 40)
print(f"[2] Banana Collected    : {final_banana} images")
print(f"    - Safe Copied Raw   : {propagated_raw_count}")
print(f"    - Crawled Generic   : {dl_banana_count}")
print("-" * 40)
print(f"[3] Not-Banana Collected: {final_not_banana} images")
print(f"    - Yellow Objects    : {dl_yellow_count}")
print(f"    - Backgrounds       : {dl_bg_count}")
print(f"    - Yellow Obj Ratio  : {yellow_ratio:.1f}%")
print("########################################")
print("All tasks completed successfully. 300x300 BGR padding & UUID naming applied.")
