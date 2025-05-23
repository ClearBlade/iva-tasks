# Scheduled Video Capture Task

- Task ID: `scheduled_recording`
- Input Topic: `task/scheduled_recording/input`
- Input Payload:
  ```json
  {
    "uuid": "daf6af50-7e90-4ab6-80cc-501e9e3afb01", // (string) Task UUID
    "task_id": "scheduled_recording", // (string) Task ID
    "camera_id": "Hikvision101", // (string) Camera ID
    "task_settings": {
      "root_path": {
        "id": "google-bucket-set", // {string} Name of your bucket set
        "path": "/tmp/clearblade_platform_buckets" // {string} The edge root directory specified by your bucket set
      },
      "file_type": "mp4", // (string) File type ["mp4", "avi"]
      "resolution": "Original", // (string) Video Quality - Original, Lower, Lowest
      "interval": 600, // (number) Interval between video start times
      "interval_units": "Seconds", // (string) Units of interval ["Seconds", "Minutes", "Hours", "Days"] Defaults to "Seconds"
      "start_time": "2025-02-25T15:31:05.423Z", // (string) Desired interval start time
      "clip_length": 10, // (integer) Desired video length
      "clip_length_units": "Seconds" // (string) Units of clip_length ["Seconds", "Minutes", "Hours"] Defaults to "Seconds"
    }
  }
  ```
- Output Topic:
  task/{TASK_ID}/output/{camera_id} where TASK_ID is 'scheduled_recording'
- Output Payload:
  ```json
  {
      **input_payload,
      "scheduled_recording_output": {
          "save_path": "/tmp/clearblade_platform_buckets/<system_key>/<camera_id>/scheduled_recording/yyyy-mm-dd/yyyy-mm-dd_hh.mm.ss.mp4" // If no saved video this cycle, save_path is None.
      }
  }
  ```
