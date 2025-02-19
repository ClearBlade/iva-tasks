import json
import os
import time
from clearblade_mqtt_library import AdapterLibrary
from dotenv import load_dotenv
from multiprocessing import shared_memory as shm
import numpy as np
import signal
import sys
from object_detection import detect_objects

TASK_ID = 'object_detection'

def handle_sigterm(signum, frame):
    global existing_mem
    print("\n[Reader] SIGTERM received. Cleaning up shared memory...")
    if existing_mem:
        existing_mem.close()  #Close but DO NOT unlink
    sys.exit(0)

def on_message(message):
    data = json.loads(message.payload.decode())
    camera_id = data.get('camera_id')
    task_uuid = data.get('uuid')
    task_settings = data.get('task_settings', {})
    frame_shape = data.get('frame_shape')
    existing_mem = shm.SharedMemory(name=f'{task_uuid}_frame')
    image = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_mem.buf)
    if not image.any():
        print('Invalid frame data')
        return
    image_with_bboxes, bboxes, objects_detected, total_objects = detect_objects(camera_id, task_settings, image, frame_shape)
    existing_mem.buf[:image_with_bboxes.nbytes] = image_with_bboxes.tobytes()
    existing_mem.close()
    data[f"{TASK_ID}_output"] = {
        "bboxes": bboxes,
        "objects_detected": objects_detected,
        "total_objects_detected": total_objects
    }
    publish_path = data.get('publish_path')
    publish_path.remove(input_topic)
    if len(publish_path) > 0:
        adapter.publish(publish_path[0], json.dumps(data))
    else:
        output_topic = f'task/{TASK_ID}/output/{camera_id}'
        adapter.publish(output_topic, json.dumps(data))
        
signal.signal(signal.SIGTERM, handle_sigterm)

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
        