import json
import os
import signal
import sys
import time
from multiprocessing import shared_memory as shm

import numpy as np
from dotenv import load_dotenv

from scheduled_recording import save_frame

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clearblade_mqtt_library import AdapterLibrary

TASK_ID = 'scheduled_recording'
existing_mem = None  #Global variable for shared memory

def handle_sigterm(signum, frame):
    global existing_mem
    print("\n[Reader] SIGTERM received. Cleaning up shared memory...")
    if existing_mem:
        existing_mem.close()  #Close but DO NOT unlink
    sys.exit(0)

def on_message(message):
    data = json.loads(message.payload.decode())
    task_uuid = data.get('uuid')
    camera_id = data.get('camera_id')
    task_settings = data.get('task_settings')
    frame_shape = data.get('frame_shape')
    frame = None
    existing_mem = None
    try:
        existing_mem = shm.SharedMemory(name=f"{task_uuid}_frame")        
        #Get the frame from shared memory
        frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_mem.buf)
        
        #Untie the frame from shared memory
        frame = frame.copy()
        
        #Close shared memory
        if existing_mem:
            existing_mem.close()

        #Check frame validity
        invalid_frame = False
        if frame is None:
            print(f'ERROR: Frame from shared memory is None')
            invalid_frame = True
        elif frame.size == 0:
            print(f'ERROR: Frame from shared memory has size 0')
            invalid_frame = True
        elif len(frame.shape) < 2:
            print(f'ERROR: Frame from shared memory has invalid shape: {frame.shape}')
            invalid_frame = True
        
        if invalid_frame:
            return
            
    except Exception as e:
        print(f"ERROR: Error accessing shared memory: {e}")
        if existing_mem:
            existing_mem.close()
        return
    
    path = save_frame(frame, camera_id, task_uuid, task_settings, TASK_ID)    
    
    if path:
        print(f'VIDEO SAVED TO: {path}')
        data[f"{TASK_ID}_output"] = {
            "saved_path": path,
        }
    else:
        data[f"{TASK_ID}_output"] = {
            "saved_path": None,
        }

    publish_path = data.get('publish_path', [])
    if input_topic in publish_path:
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

    print(f'\nListening for messages on task input topic "{input_topic}"...')

    while True:
        time.sleep(1)
