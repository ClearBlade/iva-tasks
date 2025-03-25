import json
import os
import time
from clearblade_mqtt_library import AdapterLibrary
from dotenv import load_dotenv
from multiprocessing import shared_memory as shm
import numpy as np
import signal
import sys
import cv2
from object_detection import detect_objects
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recording_utils import clear_recordings
TASK_ID = 'object_detection'
INPUT_TOPIC = f'task/{TASK_ID}/input'

last_capture_time = {}

def handle_sigterm(signum, frame):
    global existing_mem
    print("\n[Reader] SIGTERM received. Cleaning up shared and temp memory...")          
    if existing_mem:
        existing_mem.close()  #Close but DO NOT unlink
    clear_recordings(TASK_ID)
    sys.exit(0)

def on_message(message):
    global existing_mem
    data = json.loads(message.payload.decode())
    camera_id = data.get('camera_id')
    task_uuid = data.get('uuid')
    task_settings = data.get('task_settings', {})
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
    except Exception as e:
        print(f"Error processing frame: {e}")
    finally:
        #Always close the shared memory
        if 'existing_mem' in locals() and existing_mem:
            existing_mem.close()

    data[f"{TASK_ID}_output"] = {
        "bboxes": bboxes,
        "objects_detected": objects_detected,
        "total_objects_detected": total_objects
    }
    if data.get('task_id', TASK_ID) == TASK_ID: #to make sure the needs_video is meant for this task
        from recording_utils import LastCaptureTime, TIME_UNITS, SUPPORTED_IMAGE_FILE_TYPES
        event = (len(objects_detected) > 0 and total_objects > 0)
        root_path = task_settings.get('root_path', '/assets/saved_videos')
        file_type = task_settings.get('file_type', '').lower()
        needs_video = False
        needs_snapshot = False
        clip_length = int(task_settings.get('clip_length', 0) * TIME_UNITS.get(task_settings.get('clip_length_units', 'Seconds'), 1))
        if clip_length > 0:
            needs_video = True
        elif file_type in SUPPORTED_IMAGE_FILE_TYPES:
            needs_snapshot = True
        elif file_type != '':
            print(f'Unsupported file type: {file_type}')
        if needs_video:
            scheduled_video_uuid = task_uuid + '_annotated'
            #Import necessary utilities if not already imported
            from recording_utils import (
                setup_event_recording, handle_event_recording, add_to_shared_memory
            )
            add_to_shared_memory(scheduled_video_uuid, image_with_bboxes)
            recording_lead_time = int(task_settings.get('recording_lead_time', 5))
        
            #Initialize last_capture_time if not already set and clear recordings if needed
            if camera_id not in last_capture_time:
                last_capture_time[camera_id] = LastCaptureTime()
                last_capture_time[camera_id].last_video_capture = time.time() - clip_length
                clear_recordings(TASK_ID, camera_id)
        
            file_type = 'mp4'
        
            #Setup event recording (initializes cache and temp directories if needed)
            cache_base_path = setup_event_recording(
                camera_id,
                scheduled_video_uuid,
                recording_lead_time,
                clip_length,
                adapter,
                TASK_ID,
                task_settings.get("resolution", "Low"),
                frame_shape
            )
        
            #Handle the event recording logic
            last_capture_time[camera_id].last_video_capture = handle_event_recording(
                camera_id,
                event,
                last_capture_time[camera_id].last_video_capture,
                recording_lead_time,
                clip_length,
                cache_base_path,
                root_path,
                file_type,
                TASK_ID,
                data[f"{TASK_ID}_output"]  #To update with the saved_video_path
            )
        elif needs_snapshot:
            retrigger_delay = int(task_settings.get('retrigger_delay', 15) * TIME_UNITS.get(task_settings.get('retrigger_delay_units', 'Seconds'), 1))
            if camera_id not in last_capture_time:
                last_capture_time[camera_id] = LastCaptureTime()
                last_capture_time[camera_id].last_snapshot = time.time()-retrigger_delay
            if (len(objects_detected) > 0 and total_objects > 0) and (time.time() - last_capture_time[camera_id].last_snapshot >= retrigger_delay):
                last_capture_time[camera_id].last_snapshot = time.time()
                timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
                subfolder = timestamp.split('_')[0]
                subfolder = timestamp.split('_')[0]
                system_key = os.environ.get('CB_SYSTEM_KEY', 'default_system')
                if not os.path.exists(f'{root_path}/{system_key}/{camera_id}/{TASK_ID}/{subfolder}'):
                    os.makedirs(f'{root_path}/{system_key}/{camera_id}/{TASK_ID}/{subfolder}')
                file_type = task_settings.get('file_type', 'png').lower()
                if file_type not in SUPPORTED_IMAGE_FILE_TYPES:
                    file_type = 'png'
                cv2.imwrite(f'{root_path}/{system_key}/{camera_id}/{TASK_ID}/{subfolder}/{timestamp}.{file_type}', image)
                print(f'snapshot saved to {root_path}/{system_key}/{camera_id}/{TASK_ID}/{subfolder}/{timestamp}.{file_type}')
    publish_path = data.get('publish_path')
    if len(publish_path) > 1:
        publish_path.remove(INPUT_TOPIC)
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
    adapter.subscribe(INPUT_TOPIC, on_message)
    print(f'Listening for messages on task input topic "{INPUT_TOPIC}"...')
    while True:
        time.sleep(1)
        