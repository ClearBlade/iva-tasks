import os
from datetime import datetime, timezone

import cv2
import dateutil.parser as parser

# Save path will be /tmp/clearblade_platform_buckets/<system_key>/<camera_id>/<date>/<time>.jpg

last_saved_times = {}  # Global variable to track last saved second

def get_quality_perc(resolution):
    if resolution == 'High':
        return 100
    elif resolution == 'Medium':
        return 75
    return 50

def check_interval(camera_id, interval, start_time):
    global last_saved_times

    interval = int(interval)    
    start_time = parser.parse(start_time)
    current_time = datetime.now(timezone.utc)
    
    if current_time >= start_time:
        time_difference = (current_time - start_time).total_seconds()
        current_second = int(time_difference)
        if current_second % interval == 0 and current_second != last_saved_times.get(camera_id, -1):
            last_saved_times[camera_id] = current_second
            current_time = current_time.astimezone()
            return current_time.strftime("%Y-%m-%d"), current_time.strftime("%H:%M:%S")
    
    return "", ""

def save(root_path: str, frame, camera_id: str, resolution: str, file_type: str, sub_folder:str, name: str):

    frame = cv2.resize(frame, (0, 0), fx=get_quality_perc(resolution)/100, fy=get_quality_perc(resolution)/100, interpolation=cv2.INTER_AREA)

    system_key = os.environ['CB_SYSTEM_KEY']
    save_path = os.path.join(root_path, system_key, camera_id, sub_folder)
    os.makedirs(save_path, exist_ok=True)
        
    file_path = os.path.join(save_path, f"{name}.{file_type.lower()}")
    cv2.imwrite(file_path, frame)
    return file_path

def save_frame(frame, camera_id, task_settings):
    root_path = task_settings.get("root_path", "./assets/saved_frames")
    file_type = task_settings.get("file_type", "JPG")
    resolution = task_settings.get("resolution", "Low")
    interval = task_settings.get("interval", 3600)
    start_time = task_settings.get("start_time", datetime.now().isoformat())

    sub_folder, name = check_interval(camera_id, interval, start_time)
    if sub_folder and name:
        return save(root_path, frame, camera_id, resolution, file_type, sub_folder, name)
    
    return ""

if __name__ == '__main__':
    import os

    os.environ['CB_SYSTEM_KEY'] = "test_system_key"

    camera_id = "camera_1"
    frame = cv2.imread("assets/test.png")
    path = save_frame(frame, camera_id, {
        "root_path": "assets/saved_frames",
        "file_type": "PNG",
        "resolution": 'High',
        "interval": "10", # in seconds
        "start_time": "2025-02-25T15:31:05.423Z",
    })
    print('saved image at:', path)
    