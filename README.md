# iva-tasks
This repository stores the source code for all the IVA tasks

## Steps for creating a new IVA task

- Create a folder with your task ID. Your task ID must be snake-cased. For eg: object_detection, line_crossing, etc.
- Every task folder must have the same files hierarchy as follows:
  - task.py - Microservice for handling task related functioning (custom for each task)
  - mqtt_listener.py - Service that listens to a topic and sends data to task.py
    - Update the `TASK_ID` to match your folder name
    - Input topic should always be `task/{TASK_ID}/input`
    - The input data payload will have the following structure
    ```
    {
        "uuid": <task_uuid>,              // UUID of the task
        "frame": <base64encoded_image>,   // Base64 encoded image string
        "camera_id": <camera_id>,         // Camera ID
        "task_settings": {},              // Task settings specific to the task set from the UI
        "object_detection_output": {},    // Object Detection Output (if task needs object detection)
        "object_tracking_output": {}      // Object Tracking Output (if task needs object tracking)
    }
    ```
    - The output of your task must be added to the input data payload and must have a key `{TASK_ID}_output`. For example, `line_crossing` task's output should be added to `input_payload['line_crossing_output']`.  
    - Output topic should always be `task/{TASK_ID}/output/{CAMERA_ID}`
  - clearblade_mqtt_library.py - ClearBlade Adapter Library (No need to update this)
  - requirements.txt - List of dependencies required to run this task
  - assets - Folder to store task assets e.g. model artifacts (Optional)

## Running and testing a task locally  

- Once you have your task folder setup, you can run the task using: \
``` python -m venv ./venv && venv/bin/pip install -r requirements.txt && venv/bin/python mqtt_listener.py --platformURL=<platform-url> --messagingURL=<messaging-url> ```
