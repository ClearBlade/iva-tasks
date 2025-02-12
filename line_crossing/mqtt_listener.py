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
        existing_mem.close()  # Close but DO NOT unlink
    sys.exit(0)

def convertB64ToFrame(b64_string):
    jpg_original = base64.b64decode(b64_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    return img

def convertFrameToB64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def on_message(message):
    start_time = time.time()
    data = json.loads(message.payload.decode())
    
    camera_id = data.get('camera_id')
    task_settings = data.get('task_settings', {})
    
    objects_to_detect = task_settings.get('objects_to_detect', [])
    object_tracking = task_settings.get('object_tracking', False)
    line = task_settings.get('line')
    
    if not objects_to_detect or not object_tracking:
        print('Invalid task settings: objects_to_detect is empty or object_tracking is false')
        return
    frame_shape = data.get('frame_shape')
    existing_mem = shm.SharedMemory(name=f"{camera_id}_frame")
    drawn_frame = np.ndarray(frame_shape, dtype=np.uint8, buffer=existing_mem.buf)

    if camera_id not in camera_trackers:
        #construct with initial frame and frame_shape to choose the best line color
        camera_trackers[camera_id] = CameraTracker(drawn_frame, frame_shape, line)
    
    detection_output = data.get('object_detection_output', {})
    #print('Detection output:', detection_output)
    direction = task_settings.get('direction', None)
    box_data = detection_output.get('bboxes', {})
    
    #Set direction to None if it's not A_TO_B or B_TO_A
    if direction not in [DIRECTION_A_TO_B, DIRECTION_B_TO_A]:
        direction = None
    
    camera_trackers[camera_id].update(box_data, line)
    
    results = camera_trackers[camera_id].process_crossings(objects_to_detect, direction)
    #drawn_frame = convertB64ToFrame(data.get('frame'))
    drawn_frame_with_line = camera_trackers[camera_id].draw_line(drawn_frame)
    #timestamp = time.strftime("%Y%m%d-%H%M%S") + str(time.time() % 1)[1:3]
    #cv2.imwrite(f"{timestamp}_drawn_frame_with_line.jpg", drawn_frame_with_line)
    existing_mem.buf[:drawn_frame_with_line.nbytes] = drawn_frame_with_line.tobytes()
    existing_mem.close()
    if any(results.values()):
        for classification, crossing in results.items():
            if crossing and (direction is None or crossing == direction):
                data[f"{TASK_ID}_output"] = {
                    "direction": crossing,
                    "crossing": True,
                    "classification": classification,
                }
                
                #Uncomment the lines below to save the annotated frame to a file
                frame_name = time.strftime("%Y%m%d-%H%M%S") + str(time.time() % 1)[1:3]
                #cv2.imwrite(f"{frame_name}.jpg", drawn_frame_with_line)
                #print('CROSSING DETECTED!!!!!!!!\n Image saved as:', f"{timestamp}_drawn_frame_with_line.jpg")              
                output_topic = data.get('publish_path', [f'task/{TASK_ID}/output/{camera_id}'])[0]
                data['publish_path'].remove(f'task/{TASK_ID}/input')
                adapter.publish(output_topic, json.dumps(data))
    else: #No crossings detected
        # data["frame"] = convertFrameToB64(convertB64ToFrame(data.get('frame')))
        output_topic = f'task/{TASK_ID}/output/{camera_id}'
        data[f"{TASK_ID}_output"] = {
            "direction": None,
            "crossing": False,
            "classification": None,
        }
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        #cv2.imwrite(f'{timestamp}_person_no_crossing_detected.jpg', drawn_frame_with_line)
        adapter.publish(output_topic, json.dumps(data))
        #print('No crossings detected object saved as', f"{timestamp}_person_no_crossing_detected.jpg")
    #print(f"Time to process: {time.time() - start_time:.6f} seconds")
    
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