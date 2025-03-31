# Scheduled Snapshot Task

- Task ID: `scheduled_snapshot`
- Input Topic: `task/scheduled_snapshot/input`
- Input Payload: 
    ```json
    {
        "uuid": "daf6af50-7e90-4ab6-80cc-501e9e3afb01",     // (string) Task UUID 
        "frame": "",                                        // (string) Base64 encoded image string
        "camera_id": "Hikvision101",                        // (string) Camera ID
        "task_settings": {  
            "root_path": {
                "id": "google-bucket-set",               // {string} Name of your bucket set
                "path": "/tmp/clearblade_platform_buckets"  // {string} The edge root directory specified by your bucket set
            },
            "file_type": "PNG",                                 // (string) File type - PNG, JPG, JPEG
            "resolution": "Original",                           // (string) Image Quality - Original, Lower, Lowest
            "interval": 600,                                    // (number) Snapshot Interval (secs)
            "start_time": "2025-02-25T15:31:05.423Z",           // (string) Snapshot start time
        }, 
    }
    ```
- Output Topic: 
    task/{TASK_ID}/output/{camera_id} where TASK_ID is 'scheduled_snapshot' 
- Output Payload:
    ```json
    {
        **input_payload,
        "scheduled_snapshot_output": {
            "save_path": "/tmp/clearblade_platform_buckets/<system_key>/<camera_id>/<date>/<time>.jpg"
        }   
    }
    ```
