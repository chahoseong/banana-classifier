import requests
import time
from pathlib import Path

# Paths to test
project_root = Path(__file__).resolve().parent
test_images_dir = project_root / 'dataset' / 'raw'

def test_predict_endpoint(image_path):
    print(f"\n--- Testing Endpoint with {image_path.name} ---")
    url = "http://127.0.0.1:8000/predict"
    
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/jpeg')}
        
        start_time = time.time()
        try:
            response = requests.post(url, files=files)
            end_time = time.time()
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.json()}")
            print(f"Total Latency (Client-side): {(end_time - start_time) * 1000:.2f} ms")
        except Exception as e:
            print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    # Give the server a second to start
    time.sleep(2)
    
    # Let's find a few test images
    test_files = []
    if test_images_dir.exists():
        for i, filepath in enumerate(test_images_dir.glob('**/*.jpg')):
            if filepath.is_file():
                test_files.append(filepath)
            if len(test_files) >= 3:
                break
                
    if not test_files:
        print("No test images found in dataset/raw directory.")
    else:
        for f in test_files:
            test_predict_endpoint(f)
