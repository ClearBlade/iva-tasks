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
            "roi_coords": [],                                 // (number[]) Coordinates of region of interest (optional)
            "confidence_threshold": 50                        // (number) Confidence threshold of the model (optional; default = 50%)
        }, 
    }
    ```
- Output Topic: `task/object_detection/output/<camera_id>`
- Output Payload:
    ```json
    {
        **input_payload,
        "object_detection_output": {                         // key = object class & value = bounded boxes 
            "person": [
                [553.8050537109375, 93.83261108398438, 608.624267578125, 199.79129028320312],
                [327.7948913574219, 160.7405242919922, 351.0251159667969, 182.00564575195312]
            ],
            "boat": [
                [192.2393341064453, 160.13238525390625, 411.2384033203125, 184.743408203125]
            ]
        }   
    }
    ```
