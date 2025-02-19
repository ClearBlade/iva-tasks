import json
import time
import cv2
import numpy as np
import base64
from multiprocessing import shared_memory as shm
import numpy as np
import signal
import sys

from clearblade_mqtt_library import AdapterLibrary
from dotenv import load_dotenv
from line_crossing import CameraTracker, DIRECTION_A_TO_B, DIRECTION_B_TO_A

TASK_ID = 'line_crossing'

camera_trackers = {}

def handle_sigterm(signum, frame):
    global existing_mem
    print("\n[Reader] SIGTERM received. Cleaning up shared memory...")
    if existing_mem:
        existing_mem.close()  #Close but DO NOT unlink
    sys.exit(0)

def on_message(message):
    data = json.loads(message.payload.decode())
    camera_id = data.get('camera_id')
    task_settings = data.get('task_settings', {})
    task_uuid = data.get('uuid')
    objects = task_settings.get('objects_to_detect', [])
    objects_to_detect = [obj for obj, settings in objects.items() if settings.get('enable_tracking', False)]
    line = task_settings.get('line')
    if len(objects_to_detect) == 0:
        print('Invalid task settings: No objects have tracking enabled')
        return
    frame_shape = data.get('frame_shape')
    existing_mem = shm.SharedMemory(name=f"{task_uuid}_frame")
    drawn_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_mem.buf)
    if camera_id not in camera_trackers:
        camera_trackers[camera_id] = CameraTracker(drawn_frame, frame_shape, line)
    detection_output = data.get('object_detection_output', {})
    direction = task_settings.get('direction', None)
    box_data = detection_output.get('bboxes', {})
    #Set direction to None if it's not A_TO_B or B_TO_A
    #Allows for detection of all crossings regardless of direction
    if direction not in [DIRECTION_A_TO_B, DIRECTION_B_TO_A]:
        direction = None
    camera_trackers[camera_id].update(box_data, line)
    results = camera_trackers[camera_id].process_crossings(objects_to_detect, direction)
    drawn_frame_with_line = camera_trackers[camera_id].draw_line(drawn_frame)
    if any(results.values()):
        for classification, crossing in results.items():
            if crossing and (direction is None or crossing == direction):
                data[f"{TASK_ID}_output"] = {
                    "direction": crossing,
                    "crossing": True,
                    "classification": classification,
                }
                output_topic = data.get('publish_path', [f'task/{TASK_ID}/output/{camera_id}'])[0]
                data['publish_path'].remove(f'task/{TASK_ID}/input')
                existing_mem.buf[:drawn_frame_with_line.nbytes] = drawn_frame_with_line.tobytes()
                adapter.publish(output_topic, json.dumps(data))
    else: #No crossings detected
        output_topic = f'task/{TASK_ID}/output/{camera_id}'
        data[f"{TASK_ID}_output"] = {
            "direction": None,
            "crossing": False,
            "classification": None,
        }
        existing_mem.buf[:drawn_frame_with_line.nbytes] = drawn_frame_with_line.tobytes()
        adapter.publish(output_topic, json.dumps(data))
    existing_mem.close()
    
signal.signal(signal.SIGTERM, handle_sigterm)

if __name__ == '__main__':   
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