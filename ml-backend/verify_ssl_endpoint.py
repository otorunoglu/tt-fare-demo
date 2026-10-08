import requests
import time
import sys

BASE_URL = "http://localhost:30040"


def test_ssl_train():
    print("Testing SSL Train endpoint...")
    payload = {
        "project_id": 1,
        "stable_id": "stable03",
        "stall_id": "all",
        "date_from": "2025-12-01",
        "date_until": "2025-12-30",  # Smaller range to avoid timeout
        "epochs": 1,  # fast test
        "batch_size": 4,
    }

    try:
        resp = requests.post(f"{BASE_URL}/api/smartstable/ssl/train", json=payload)
        print(f"Status Code: {resp.status_code}")
        print(f"Response: {resp.json()}")

        if resp.status_code == 200:
            return True
        else:
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False


def test_ssl_status():
    print("\nChecking SSL Status...")
    for i in range(10):
        resp = requests.get(f"{BASE_URL}/api/smartstable/ssl/status")
        status = resp.json()
        print(f"Job Status: {status}")
        if status.get("status") in ["completed", "failed"]:
            return status.get("status")
        time.sleep(2)
    return "timeout"


def test_ssl_visualize():
    print("\nTesting SSL Visualize endpoint...")
    payload = {
        "project_id": 1,
        "stable_id": "stable03",
        "stall_id": "all",
        "date_from": "2024-01-01",
        "date_until": "2025-12-31",
        "max_samples": 10,
    }
    resp = requests.post(f"{BASE_URL}/api/smartstable/ssl/visualize", json=payload)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Image received (length): {len(data.get('image_base64', ''))}")
        return True
    else:
        print(f"Response: {resp.text}")
        return False


if __name__ == "__main__":
    if test_ssl_train():
        status = test_ssl_status()
        if status == "completed":
            test_ssl_visualize()
        else:
            print("Skipping visualization due to training failure/timeout")
