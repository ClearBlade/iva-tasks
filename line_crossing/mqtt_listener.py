import json
import time
import cv2
import numpy as np
from multiprocessing import shared_memory as shm
import numpy as np
import signal
import sys

from clearblade_mqtt_library import AdapterLibrary
from dotenv import load_dotenv
from line_crossing import CameraTracker, DIRECTION_A_TO_B, DIRECTION_B_TO_A
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recording_utils import clear_recordings

TASK_ID = 'line_crossing'
INPUT_TOPIC = f'task/{TASK_ID}/input'

last_capture_time = {}

camera_trackers = {}

def handle_sigterm(signum, frame):
    global existing_mem
    print("\n[Reader] SIGTERM received. Cleaning up shared and temp memory...")          
    if existing_mem:
        existing_mem.close()  #Close but DO NOT unlink
    clear_recordings(TASK_ID)
    sys.exit(0)

def on_message(message):
    data = json.loads(message.payload.decode())
    camera_id = data.get('camera_id')
    task_settings = data.get('task_settings', {})
    task_uuid = data.get('uuid')
    objects = task_settings.get('objects_to_detect', [])
    objects_to_detect = [obj for obj, settings in objects.items() if settings.get('enable_tracking', False)]
    x1, y1, x2, y2 = task_settings.get('line')
    line = [[x1, y1], [x2, y2]]
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
    outputs = data.get('publish_path', [f'task/{TASK_ID}/output/{camera_id}'])
    if len(outputs) > 1:
        if INPUT_TOPIC in outputs:
            data['publish_path'].remove(INPUT_TOPIC)
        output_topic = outputs[0]
    else:
        output_topic = f'task/{TASK_ID}/output/{camera_id}'
    event = False
    if any(results.values()):
        for classification, crossing in results.items():
            if crossing and (direction is None or crossing == direction):
                if isinstance(crossing, list):
                    crossing = crossing[0]
                data[f"{TASK_ID}_output"] = {
                    "direction": crossing,
                    "crossing": True,
                    "classification": classification,
                    "message": task_settings.get(crossing, 'Crossing detected')
                }
                event = True          
                existing_mem.buf[:drawn_frame_with_line.nbytes] = drawn_frame_with_line.tobytes()
                adapter.publish(output_topic, json.dumps(data))
    else: #No crossings detected
        data[f"{TASK_ID}_output"] = {
            "direction": None,
            "crossing": False,
            "classification": None,
        }
        existing_mem.buf[:drawn_frame_with_line.nbytes] = drawn_frame_with_line.tobytes()
        adapter.publish(output_topic, json.dumps(data))
    
    if data.get('task_id', TASK_ID) == TASK_ID: #to make sure the needs_video/snapshot is meant for this task
        from recording_utils import LastCaptureTime, time_units
        root_path = task_settings.get('root_path', './assets/saved_videos')
        if task_settings.get('needs_video', False):
            scheduled_video_uuid = task_uuid + '_annotated'
            from recording_utils import (
                setup_event_recording, handle_event_recording, add_to_shared_memory, supported_video_file_types
            )
            add_to_shared_memory(scheduled_video_uuid, drawn_frame_with_line)
            #Get the clip_length and clip_length_units from task_settings
            clip_length = int(task_settings.get('clip_length', 15) * time_units.get(task_settings.get('clip_length_units', 'Seconds'), 1))
            #Get recording_lead_time from task_settings in seconds
            recording_lead_time = int(task_settings.get('recording_lead_time', 5))
        
            #Initialize last_capture_time if not already set
            if camera_id not in last_capture_time:
                last_capture_time[camera_id] = LastCaptureTime()
                last_capture_time[camera_id].last_video_capture = time.time() - clip_length
                clear_recordings(TASK_ID, camera_id)
        
            #Get other settings
            file_type = task_settings.get('file_type', 'mp4').lower()
            if file_type not in supported_video_file_types:
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
        elif task_settings.get('needs_snapshot', False):
            from recording_utils import supported_image_file_types
            retrigger_delay = int(task_settings.get('retrigger_delay', 5) * time_units.get(task_settings.get('retrigger_delay_units', 'Seconds'), 1))
            if camera_id not in last_capture_time:
                last_capture_time[camera_id] = LastCaptureTime()
                last_capture_time[camera_id].last_snapshot = time.time()-retrigger_delay
            if event and (time.time() - last_capture_time[camera_id].last_snapshot >= retrigger_delay):
                last_capture_time[camera_id].last_snapshot = time.time()
                timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
                subfolder = timestamp.split('_')[0]
                system_key = os.environ.get('CB_SYSTEM_KEY', 'default_system')
                if not os.path.exists(f'{root_path}/{system_key}/{camera_id}/{TASK_ID}/{subfolder}'):
                    os.makedirs(f'{root_path}/{system_key}/{camera_id}/{TASK_ID}/{subfolder}')
                file_type = task_settings.get('file_type', 'png').lower()
                if file_type not in supported_image_file_types:
                    file_type = 'png'
                cv2.imwrite(f'{root_path}/{camera_id}/{TASK_ID}/{subfolder}/{timestamp}.{file_type}', drawn_frame_with_line)
                print('snapshot saved')
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

    adapter.subscribe(INPUT_TOPIC, on_message)
    print(f'Listening for messages on task input topic "{INPUT_TOPIC}"...')

    while True:
        time.sleep(1)
