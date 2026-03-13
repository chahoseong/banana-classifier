import requests
import time
import subprocess
import sys
import os

def check_server():
    print("Testing connectivity to http://localhost:8000...")
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        print(f"Server responded with: {response.status_code} - {response.json()}")
        return True
    except Exception as e:
        print(f"Failed to connect: {e}")
        return False

if __name__ == "__main__":
    if check_server():
        print("Backend seems to be running.")
    else:
        print("Backend is NOT responding. Attempting to start it for diagnostics...")
        # This is just a test, won't keep it running
        env = os.environ.copy()
        env["PYTHONPATH"] = ".."
        p = subprocess.Popen([sys.executable, "main.py"], cwd=".", env=env)
        time.sleep(5) # Wait for startup
        if check_server():
            print("Backend started successfully in background.")
            p.terminate()
        else:
            print("Backend failed to start even in background.")
            p.terminate()
