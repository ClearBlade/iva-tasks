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
from line_crossing import rescale_line, CameraTracker, DIRECTION_A_TO_B, DIRECTION_B_TO_A, UI_SCALE
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recording_utils import clear_recordings

TASK_ID = 'line_crossing'
INPUT_TOPIC = f'task/{TASK_ID}/input'

last_capture_time = {}

camera_trackers = {}

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
    task_settings = data.get('task_settings', {})
    task_uuid = data.get('uuid')
    capture_key = f"{camera_id}_{task_uuid}"
    objects = task_settings.get('objects_to_detect', [])
    objects_to_detect = [obj for obj, settings in objects.items() if settings.get('enable_tracking', False)]
    frame_shape = data.get('frame_shape')
    scaled_line = rescale_line(task_settings.get('line'), frame_shape, UI_SCALE)
    line = [[scaled_line[0], scaled_line[1]], [scaled_line[2], scaled_line[3]]]
    
    if len(objects_to_detect) == 0:
        print('Invalid task settings: No objects have tracking enabled')
        return
        
    frame_shape = data.get('frame_shape')
    
    try:    
        existing_mem = shm.SharedMemory(name=f"{task_uuid}_frame")
        drawn_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_mem.buf)
        drawn_frame_copy = drawn_frame.copy()
        
        if capture_key not in camera_trackers:
            camera_trackers[capture_key] = CameraTracker(drawn_frame_copy, frame_shape, line)
            
        detection_output = data.get('object_detection_output', {})
        direction = task_settings.get('direction', None)
        box_data = detection_output.get('bboxes', {})
        
        #Set direction to None if it's not A_TO_B or B_TO_A
        #Allows for detection of all crossings regardless of direction
        if direction not in [DIRECTION_A_TO_B, DIRECTION_B_TO_A]:
            direction = None
            
        camera_trackers[capture_key].update(box_data, line)
        results = camera_trackers[capture_key].process_crossings(objects_to_detect, direction)
        drawn_frame_with_line = camera_trackers[capture_key].draw_line(drawn_frame_copy)
        
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
                        "direction": task_settings.get(crossing, 'Crossing detected'),
                        "crossing": True,
                        "classification": classification,
                    }
                    event = True          
                    existing_mem.buf[:drawn_frame_with_line.nbytes] = drawn_frame_with_line.tobytes()
        else: #No crossings detected
            data[f"{TASK_ID}_output"] = {
                "direction": None,
                "crossing": False,
                "classification": None,
                "saved_path": None
            }
            existing_mem.buf[:drawn_frame_with_line.nbytes] = drawn_frame_with_line.tobytes()
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        
    finally:
        #Always close the shared memory
        if 'existing_mem' in locals() and existing_mem:
            existing_mem.close()
    
    if data.get('task_id', TASK_ID) == TASK_ID:
        from recording_utils import LastCaptureTime, process_task_settings, handle_snapshot_recording, adjust_resolution
        
        settings = process_task_settings(task_settings)
        
        if capture_key not in last_capture_time:
            last_capture_time[capture_key] = LastCaptureTime()
            last_capture_time[capture_key].last_video_capture = time.time() - settings['clip_length']
            clear_recordings(TASK_ID, task_uuid, camera_id)
        
        if settings['needs_video']:
            scheduled_video_uuid = task_uuid + '_annotated'
            
            from recording_utils import (
                setup_event_recording, handle_event_recording, add_to_shared_memory
            )
            
            add_to_shared_memory(scheduled_video_uuid, drawn_frame_with_line)
            
            cache_base_path = setup_event_recording(
                camera_id,
                scheduled_video_uuid,
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
                task_uuid,
                data.get(f"{TASK_ID}_output", {})
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
                cv2.imwrite(saved_path, adjust_resolution(drawn_frame_with_line, settings['resolution']))
                print(f'snapshot saved to {saved_path}')
                data[f"{TASK_ID}_output"]["saved_path"] = saved_path
                
        elif settings['file_type'] != '':
            print(f'Unsupported file type: {settings["file_type"]}')
        
    adapter.publish(output_topic, json.dumps(data))
    
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
