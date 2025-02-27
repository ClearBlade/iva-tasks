# Object Detection Task

- Task ID: `object_detection`
- Input Topic: `task/object_detection/input`
- Input Payload: 
    ```json
    {
        "uuid": "daf6af50-7e90-4ab6-80cc-501e9e3afb01",     // (string) Task UUID 
        "frame": "",                                        // (string) Base64 encoded image string
        "camera_id": "Hikvision101",                        // (string) Camera ID
        "task_settings": {  
            "objects_to_detect": ["people", "car"],           // (string[]) Objects to detect (see assets/coco.names)
            "object_tracking": True,                          // (boolean) Enables Object Tracking (default = True)
            "confidence_threshold": 65                        // (number) Confidence threshold of the model (optional; default = 65%)
        }, 
    }
    ```
- Output Topic: 
    The output topic will be dynamic based on the next task assigned in the publish path.
    If object detection is the final or only task, it will be task/{TASK_ID}/output/{camera_id} where TASK_ID is 'object_detection'
- Output Payload with tracking:
    ```json
    {
        **input_payload,
        "frame": "123", // This will contain the frame capture buy the camera with the tracking label (ie person1) and bounding box drawn onto each detection
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
        "frame": "123", // This will contain the frame capture buy the camera with the object classification (ie person) and bounding box drawn onto each detection
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
            "total_objects_detected": 3 // total number of detected objects in frame
        }   
    }
    ```
