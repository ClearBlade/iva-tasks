import json
import os
import time

from clearblade_mqtt_library import AdapterLibrary
from dotenv import load_dotenv
from task import convertB64ToFrame, convertFrameToB64, detect_objects

TASK_ID = 'object_detection'

def on_message(message):
    data = json.loads(message.payload.decode())
    print('received message')
    frame = data.get('frame')

    if not frame:
        print('No frame found in the message')
        return

    camera_id = data.get('camera_id')
    task_settings = data.get('task_settings')
    
    image, bboxes = detect_objects(convertB64ToFrame(frame), task_settings)
    data["frame"] = convertFrameToB64(image) # replacs the input frame with the new output frame that has the detected objects
    
    count = 0
    for obj in bboxes.items():
        count += len(obj[1])

    data[f"{TASK_ID}_output"] = {
        "bboxes": bboxes, # box points of the detected objects
        "objects_detected": [obj[0] for obj in bboxes.items()],
        "total_objects_detected": count
    }

    print('bbox: ', data[f"{TASK_ID}_output"])
    
    adapter.publish(f'task/{TASK_ID}/output/{camera_id}', json.dumps(data))
    print('published message: ', bboxes)


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

    