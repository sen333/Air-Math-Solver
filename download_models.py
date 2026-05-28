import os
import urllib.request

def download_file(url, target_path):
    print(f"Downloading {url} ...")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urllib.request.urlretrieve(url, target_path)
    print(f"Saved to {target_path}")

if __name__ == "__main__":
    task_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    target = os.path.join("models", "hand_landmarker.task")
    
    if not os.path.exists(target):
        download_file(task_url, target)
    else:
        print(f"{target} already exists.")
