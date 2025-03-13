# Scheduled Video Capture Task

- Task ID: `scheduled_video_capture`
- Input Topic: `task/scheduled_video_capture/input`
- Input Payload: 
    ```json
    {
        "uuid": "daf6af50-7e90-4ab6-80cc-501e9e3afb01",     // (string) Task UUID 
        "camera_id": "Hikvision101",                        // (string) Camera ID
        "task_settings": {  
            "root_path": "/tmp/clearblade_platform_buckets/",   // (string) root path to store image
            "file_type": "mp4",                                 // (string) File type ["mp4", "avi"]
            "resolution": "Original",                           // (string) Image Quality - Original, Lower, Lowest
            "interval": 600,                                    // (number) Interval between video start times
            "interval_units": "Seconds",                        // (string) Units of interval ["Seconds", "Minutes", "Hours", "Days"] Defaults to "Seconds"
            "start_time": "2025-02-25T15:31:05.423Z",           // (string) Desired interval start time
            "clip_length": 10,                                  // (integer) Desired video length
            "clip_length_units": "Seconds",                     // (string) Units of clip_length ["Seconds", "Minutes", "Hours"] Defaults to "Seconds"
        }, 
    }
    ```
- Output Topic: 
    task/{TASK_ID}/output/{camera_id} where TASK_ID is 'scheduled_video_capture' 
- Output Payload:
    ```json
    {
        **input_payload,
        "scheduled_video_capture_output": {
            "save_path": "/tmp/clearblade_platform_buckets/<system_key>/<camera_id>/scheduled_video_capture/yyyy-mm-dd/yyyy-mm-dd_hh.mm.ss.mp4" // If no saved video this cycle, save_path is None.
        }   
    }
    ```
