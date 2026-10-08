import requests
import json


def verify_add_task():
    url = "http://localhost:30040/api/smartstable/tasks/add-single"
    payload = {
        "project_id": 1,
        "dry_run": True,
        "recorded_date": "2023-10-27",
        "recorded_time": "12:00:00",
        "audio_directory": "/label-studio/data/audio/",
        "video_directory": "/label-studio/data/video/",
        "audio_file_name": "HorseRecordingsRODE48kHz.wav",
        "video_file_name": "",
        "source": "recording",
        "horse": "daily_check",
        "report": "Verification test",
        "extra_metadata": {
            "custom_field_1": "test_value",
            "diet_type": "Hay",
            "protocol_step": "Step 1",
        },
    }

    print(f"Sending POST request to {url} with payload:")
    print(json.dumps(payload, indent=2))

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("\nResponse Status Code:", response.status_code)

        data = response.json()
        print("Response Body:")
        print(json.dumps(data, indent=2))

        task_data = data.get("task_data", {})

        # Verify custom fields are merged
        if (
            task_data.get("custom_field_1") == "test_value"
            and task_data.get("diet_type") == "Hay"
            and task_data.get("protocol_step") == "Step 1"
        ):
            print("\nVerification SUCCESS: extra_metadata merged correctly.")
        else:
            print("\nVerification FAILED: extra_metadata not merged correctly.")
            print(f"Got: {json.dumps(task_data, indent=2)}")

    except requests.exceptions.RequestException as e:
        print(f"\nVerification FAILED: Request error: {e}")
        if hasattr(e, "response") and e.response is not None:
            print("Response content:", e.response.text)


if __name__ == "__main__":
    verify_add_task()
