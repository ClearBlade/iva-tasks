import json
import os
import signal
import sys
import time
from multiprocessing import shared_memory as shm

import numpy as np
from clearblade_mqtt_library import AdapterLibrary
from dotenv import load_dotenv
from scheduled_video_capture import save_frame

TASK_ID = 'scheduled_video_capture'
existing_mem = None  # Global variable for shared memory

def handle_sigterm(signum, frame):
    global existing_mem
    print("\n[Reader] SIGTERM received. Cleaning up shared memory...")
    if existing_mem:
        existing_mem.close()  # Close but DO NOT unlink
    sys.exit(0)

def on_message(message):
    start_time = time.time()
    data = json.loads(message.payload.decode())

    task_uuid = data.get('uuid')
    camera_id = data.get('camera_id')
    task_settings = data.get('task_settings')
    frame_shape = data.get('frame_shape')
    frame = None
    existing_mem = None
    try:
        existing_mem = shm.SharedMemory(name=f"{task_uuid}_frame")        
        #Create a copy of the frame from shared memory
        frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_mem.buf)
        
        #Check frame validity
        if frame is None:
            print(f'ERROR: Frame from shared memory is None')
        elif frame.size == 0:
            print(f'ERROR: Frame from shared memory has size 0')
        elif len(frame.shape) < 2:
            print(f'ERROR: Frame from shared memory has invalid shape: {frame.shape}')
        else:
            frame = frame.copy()
            
    except Exception as e:
        print(f"ERROR: Error accessing shared memory: {e}")
        if existing_mem:
            existing_mem.close()
        return

    if frame is None or frame.size == 0:
        print(f'ERROR: No valid frame found in the message')
        if existing_mem:
            existing_mem.close()
        return
    
    # Close shared memory before processing to avoid locking it
    if existing_mem:
        existing_mem.close()
    
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
    
    end_time = time.time()
    processing_time = end_time - start_time
    print(f'Total processing time: {processing_time:.6f} seconds')

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
