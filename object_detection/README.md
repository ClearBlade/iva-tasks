# Object Detection Task

- Task ID: `object_detection`
- Input Topic: `task/object_detection/input`
- Input Payload: 
    ```json
    {
        "uuid": "daf6af50-7e90-4ab6-80cc-501e9e3afb01",     // (string) Task UUID
        "task_id": "object_detection",                      // (string) Task ID
        "camera_id": "Hikvision101",                        // (string) Camera ID
        "task_settings": {  
            "objects_to_detect": {
                "person": {                                 // {string} object classification (found in ./assets/coco.names)
                    "enable_tracking": True,                // (boolean) Enables Object Tracking (default = True)
                    "enable_blur": True,                    // {boolean} Enables blur (only supported for class 'person' for facial blur)
                    "show_boxes": True,                     // {boolean} Show bounding boxes in frame annotation
                    "show_labels": True,                    // {boolean} Show labels (e.g. Person1) in frame annotation
                    "confidence_threshold": 0.4,            // {float} Confidence Threshold of the model from 0-1
                },
                "car": {
                    "enable_tracking": True,
                    "enable_blur": True,
                    "show_boxes": True,
                    "show_labels": True,
                    "confidence_threshold": 0.4,
                },
            },
            "file_type": "mp4",                             // {string} File type of image ["jpg", "png"] used to determine if image saving is needed for task
            "recording_lead_time": 5,                       // {integer} Time in seconds video should start before object is detected.
            "clip_length": 15,                              // {integer} Desired duration of saved video (used to determine if video saving is needed for task)
            "retrigger_delay": 3,                           // {integer} Minimum time between saved snapshots
            "clip_length_units": "Seconds",                 // {string} Units of clip_length value. Accepts ["Seconds", "Minutes", "Hours", "Days"]. Defaults to "Seconds"
            "retrigger_delay_units": "Minutes",             // {string} Units of retrigger_delay value
            "root_path": {
                "id": "your-bucket-set-name",               // {string} Name of your bucket set or parent task id
                "path": "/tmp/clearblade_platform_buckets"  // {string} The local root directory specified by your bucket set or target directory
            },
            "resolution": "Original"                        // {string} Desired resolution of the video ["Original", "Lower", "Lowest"]
        }
    } 
    ```
- Output Topic: 
    The output topic will be dynamic based on the next task assigned in the publish path.
    If object detection is the final or only task, it will be task/{TASK_ID}/output/{camera_id} where TASK_ID is 'object_detection'
- Output Payload with tracking:
    ```json
    {
        **input_payload,
        "object_detection_output": {                         // key = object class & value = bounded boxes 
            "bboxes": {
                "person1": [553.8050537109375, 93.83261108398438, 608.624267578125, 199.79129028320312],
                "person2": [327.7948913574219, 160.7405242919922, 351.0251159667969, 182.00564575195312],
                "car1": [192.2393341064453, 160.13238525390625, 411.2384033203125, 184.743408203125]
            },
            "objects_detected": ["person", "car"], // all classifications detected in frame
            "total_objects_detected": 3 // total number of detected objects in frame
        }   
    }
    ```

- Output Payload with no tracking:
    ```json
    {
        **input_payload,
        "object_detection_output": {                         // key = object class & value = bounded boxes 
            "bboxes": {
                "person": [
                    [553.8050537109375, 93.83261108398438, 608.624267578125, 199.79129028320312],
                    [327.7948913574219, 160.7405242919922, 351.0251159667969, 182.00564575195312]
                ],
                "boat": [
                    [192.2393341064453, 160.13238525390625, 411.2384033203125, 184.743408203125]
                ]
            },
            "objects_detected": ["person", "boat"], // all classifications detected in frame
            "total_objects_detected": 3, // total number of detected objects in frame
            "saved_video_path": f"{root_path}/{SYSTEM_KEY}/{camera_id}/object_detection/yyyy-mm-dd/yyyy-mm-dd_hh.mm.ss.mp4" // if a video was saved, the path of the video is provided
        }   
    }
    ```
