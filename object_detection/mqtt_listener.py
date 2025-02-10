import json
import os
import time
from clearblade_mqtt_library import AdapterLibrary
from dotenv import load_dotenv
from ObjectDetection import convertB64ToFrame, convertFrameToB64, detect_objects

TASK_ID = 'object_detection'

def on_message(message):
    start_time = time.time()
    data = json.loads(message.payload.decode())
    print('received message')
    frame = data.get('frame')

    if not frame:
        print('No frame found in the message')
        return

    camera_id = data.get('camera_id')
    task_settings = data.get('task_settings', {})
    
    image = convertB64ToFrame(frame)
    
    inference_start = time.time()
    image_with_bboxes, bboxes, objects_detected, total_objects = detect_objects(image, camera_id, task_settings)
    inference_time = time.time() - inference_start

    
    data["frame"] = convertFrameToB64(image_with_bboxes)
    
    data[f"{TASK_ID}_output"] = {
        "bboxes": bboxes,
        "objects_detected": objects_detected,
        "total_objects_detected": total_objects
    }
    
    #adapter.publish(f'task/{TASK_ID}/output/{camera_id}', json.dumps(data))
    publish_path = data.get('publish_path')
    publish_path.remove(input_topic)
    if len(publish_path) > 0:
        adapter.publish(publish_path[0], json.dumps(data))
        print(f'published message to publish_path topic: {publish_path[0]}: ', data[f"{TASK_ID}_output"])
    else:
        output_topic = f'task/{TASK_ID}/output/{camera_id}'
        adapter.publish(output_topic, json.dumps(data))
        print(f'published message to output topic: {output_topic}: ', data[f"{TASK_ID}_output"])
    
    end_time = time.time()
    processing_time = end_time - start_time
    print('published message')
    print(f'Inference time: {inference_time:.6f} seconds')
    print(f'Total processing time: {processing_time:.6f} seconds')
    print('bbox: ', data[f"{TASK_ID}_output"])

if __name__ == '__main__':   
    if os.path.exists('../../.env'):
        load_dotenv(dotenv_path='../../.env')

    adapter = AdapterLibrary(TASK_ID)
    adapter.parse_arguments()
    adapter.initialize_clearblade()

    adapter.connect_MQTT()
    while not adapter.CONNECTED_FLAG:
        time.sleep(1)

    input_topic = f'task/{TASK_ID}/input'
    adapter.subscribe(input_topic, on_message)

    print(f'Listening for messages on task input topic "{input_topic}"...')

    while True:
        time.sleep(1)
        