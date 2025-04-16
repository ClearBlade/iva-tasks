import json
import os
import signal
import sys
import time
from multiprocessing import shared_memory as shm

import cv2
import numpy as np
from dotenv import load_dotenv

from object_detection import detect_objects

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from clearblade_mqtt_library import AdapterLibrary
from recording_utils import clear_recordings

TASK_ID = 'object_detection'
INPUT_TOPIC = f'task/{TASK_ID}/input'

last_capture_time = {}

def handle_sigterm(signum, frame):
    try:
        global existing_mem
        print("\n[Reader] SIGTERM received. Cleaning up shared and temp memory...")
        #Clean up all task-related recordings
        for cam_task_key in last_capture_time.keys():
            if '_' in cam_task_key:
                camera_id, task_uuid = cam_task_key.rsplit('_', 1)
                clear_recordings(TASK_ID, task_uuid, camera_id)          
        if 'existing_mem' in globals() and existing_mem:
            existing_mem.close()  #Close but DO NOT unlink
    except Exception as e:
        print(f"Error during cleanup: {e}")
    finally:
        sys.exit(0)

def on_message(message):
    global existing_mem
    data = json.loads(message.payload.decode())
    camera_id = data.get('camera_id')
    task_uuid = data.get('uuid')
    task_settings = data.get('task_settings', {})
    task_settings["task_uuid"] = task_uuid  #Add task_uuid to settings for use in detect_objects
    frame_shape = data.get('frame_shape')
    try:
        existing_mem = shm.SharedMemory(name=f'{task_uuid}_frame')
        image = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_mem.buf)
        
        #Create a deep copy of the image to avoid shared memory issues
        image_copy = image.copy()
        
        if not image.any():
            print(f"Error: Image from shared memory is empty")
            return
        #Process the copied image
        image_with_bboxes, bboxes, objects_detected, total_objects = detect_objects(camera_id, task_settings, image_copy, frame_shape)
        #Write back to shared memory
        existing_mem.buf[:image_with_bboxes.nbytes] = image_with_bboxes.tobytes()

        data[f"{TASK_ID}_output"] = {
            "bboxes": bboxes,
            "objects_detected": objects_detected,
            "total_objects_detected": total_objects,
            "saved_path": None
        }
        
        if data.get('task_id', TASK_ID) == TASK_ID:
            from recording_utils import (LastCaptureTime, adjust_resolution,
                                        handle_snapshot_recording,
                                        process_task_settings)
            
            event = (len(objects_detected) > 0 and total_objects > 0)
            
            settings = process_task_settings(task_settings)
            
            capture_key = f"{camera_id}_{task_uuid}"
            if capture_key not in last_capture_time:
                last_capture_time[capture_key] = LastCaptureTime()
                last_capture_time[capture_key].last_video_capture = time.time() - settings['clip_length']
                clear_recordings(TASK_ID, task_uuid, camera_id)
            
            if settings['needs_video']:
                annotated_uuid = f"{task_uuid}_annotated"
                
                from recording_utils import (add_to_shared_memory,
                                            handle_event_recording,
                                            setup_event_recording)

                #Save the annotated image to shared memory
                add_to_shared_memory(annotated_uuid, image_with_bboxes)
                
                cache_base_path = setup_event_recording(
                    camera_id,
                    annotated_uuid,
                    settings['recording_lead_time'],
                    settings['clip_length'],
                    adapter,
                    TASK_ID,
                    settings['resolution'],
                    frame_shape
                )
                
                last_capture_time[capture_key].last_video_capture = handle_event_recording(
                    camera_id,
                    event,
                    last_capture_time[capture_key].last_video_capture,
                    settings['recording_lead_time'],
                    settings['clip_length'],
                    cache_base_path,
                    settings['root_path'],
                    'mp4',
                    TASK_ID,
                    task_uuid,  #Use original task_uuid for saving the output
                    data[f"{TASK_ID}_output"]
                )
                            
            elif settings['needs_snapshot']:
                if not hasattr(last_capture_time[capture_key], 'last_snapshot') or last_capture_time[capture_key].last_snapshot is None:
                    last_capture_time[capture_key].last_snapshot = time.time() - settings['retrigger_delay']
                    
                last_capture_time[capture_key].last_snapshot, saved_path = handle_snapshot_recording(
                    camera_id,
                    event,
                    last_capture_time[capture_key].last_snapshot,
                    settings['retrigger_delay'],
                    settings['root_path'],
                    settings['file_type'],
                    TASK_ID,
                    task_uuid
                )
                
                if saved_path:
                    cv2.imwrite(saved_path, adjust_resolution(image_with_bboxes, settings['resolution']))
                    print(f'snapshot saved to {saved_path}')
                    data[f"{TASK_ID}_output"]["saved_path"] = saved_path
                    
            elif settings['file_type'] != '':
                print(f'Unsupported file type: {settings["file_type"]}')
                
        publish_path = data.get('publish_path')
        if len(publish_path) > 1:
            publish_path.remove(INPUT_TOPIC)
            adapter.publish(publish_path[0], json.dumps(data))
        else:
            print('publishing to output topic', f'task/{TASK_ID}/output/{camera_id}')
            output_topic = f'task/{TASK_ID}/output/{camera_id}'
            adapter.publish(output_topic, json.dumps(data))
            print('published message')
    except Exception as e:
        print(f"Error processing frame: {e}")
    finally:
        #Always close the shared memory
        if 'existing_mem' in locals() and existing_mem:
            existing_mem.close()
        
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
    adapter.subscribe(INPUT_TOPIC, on_message)
    print(f'Listening for messages on task input topic "{INPUT_TOPIC}"...')
    while True:
        time.sleep(1)
        