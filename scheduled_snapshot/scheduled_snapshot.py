import os
from datetime import datetime, timezone

import cv2
import dateutil.parser as parser

# Save path will be /tmp/clearblade_platform_buckets/<system_key>/<root_path['id']>/outbox/<camera_id>/<task_id>/<task_uuid>/<sub_folder>/<timestamp>.<file_type>"

last_saved_times = {}  # Global variable to track last saved second

def get_quality_perc(resolution):
    if resolution == 'Original':
        return 100
    elif resolution == 'Lower':
        return 75
    return 50

def get_time_in_seconds(interval, units):
    interval = int(interval)
    if units == 'Seconds':
        return interval
    elif units == 'Hours':
        return interval * 3600
    elif units == 'Days':
        return interval * 86400
    return interval * 60

def check_interval(camera_id, interval, start_time):
    global last_saved_times

    start_time = parser.parse(start_time)
    current_time = datetime.now(timezone.utc)
    
    if current_time >= start_time:
        time_difference = (current_time - start_time).total_seconds()
        current_second = int(time_difference)
        if current_second % interval == 0 and current_second != last_saved_times.get(camera_id, -1):
            last_saved_times[camera_id] = current_second
            current_time = current_time.astimezone()
            return current_time.strftime("%Y-%m-%d"), current_time.strftime("%Y-%m-%d_%H.%M.%S")
    
    return "", ""

def save(root_path: str, frame, camera_id: str, resolution: str, file_type: str, sub_folder:str, name: str, task_id: str, task_uuid: str):

    frame = cv2.resize(frame, (0, 0), fx=get_quality_perc(resolution)/100, fy=get_quality_perc(resolution)/100, interpolation=cv2.INTER_AREA)

    system_key = os.environ['CB_SYSTEM_KEY']
    save_path = os.path.join(root_path['path'], system_key, root_path['id'], 'outbox', camera_id, task_id, task_uuid, sub_folder)
    os.makedirs(save_path, exist_ok=True)
        
    file_path = os.path.join(save_path, f"{name}.{file_type.lower()}")
    cv2.imwrite(file_path, frame)
    return file_path

def save_frame(frame, camera_id, task_settings, task_id, task_uuid):
    root_path = task_settings.get("root_path", {"id": "default_id", "path": "./assets/videos"})
    file_type = task_settings.get("file_type", "JPG")
    resolution = task_settings.get("resolution", "Low")
    interval = task_settings.get("interval", 1)
    units = task_settings.get("units", "Minutes")
    start_time = task_settings.get("start_time", datetime.now().isoformat())

    interval_secs = get_time_in_seconds(interval, units)
    sub_folder, name = check_interval(camera_id, interval_secs, start_time)
    if sub_folder and name:
        return save(root_path, frame, camera_id, resolution, file_type, sub_folder, name, task_id, task_uuid)
    
    return ""

if __name__ == '__main__':
    import os
    import time

    os.environ['CB_SYSTEM_KEY'] = "test_system_key"
    camera_id = "camera_1"
    frame = cv2.imread("assets/test.png")

    while True:
        path = save_frame(frame, camera_id, {
            "root_path": {"id": "test_bucket_set_id", "path": "./assets/images"},
            "file_type": "PNG",
            "resolution": 'Original',
            "interval": "5",
            "units": "Seconds",
            "start_time": "2025-02-25T15:31:05.423Z",
        }, "scheduled_snapshot",
            "testSnapshotTask"
        )
        
        if path:
            print('saved image at:', path)
            break
    
        time.sleep(1)