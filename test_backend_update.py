import requests
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent

def test_predict_endpoint(image_path):
    print(f"\n--- Testing Endpoint with {image_path.name} ---")
    url = "http://127.0.0.1:8080/predict"
    
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/jpeg')}
        
        start_time = time.time()
        try:
            response = requests.post(url, files=files)
            end_time = time.time()
            
            print(f"Status Code: {response.status_code}")
            try:
                print(f"Response: {response.json()}")
            except:
                print(f"Response text: {response.text}")
            print(f"Total Latency: {(end_time - start_time) * 1000:.2f} ms")
        except Exception as e:
            print(f"Error connecting: {e}")

if __name__ == "__main__":
    banner = project_root / 'dataset' / 'binary_check' / 'banana'
    not_banner = project_root / 'dataset' / 'binary_check' / 'not_banana'
    
    b_files = list(banner.glob('*.jpg'))[:2] if banner.exists() else []
    nb_files = list(not_banner.glob('*.jpg'))[:2] if not_banner.exists() else []
    
    for f in b_files + nb_files:
        test_predict_endpoint(f)
