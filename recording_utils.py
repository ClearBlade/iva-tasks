import os
import time
import cv2
import json
import shutil
import numpy as np
from multiprocessing import shared_memory as shm
from datetime import datetime, timedelta
import glob

#Constants
DATETIME_FORMAT = '%Y-%m-%d_%H.%M.%S'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, 'recordings')
RECORDINGS_CACHE_DIR = os.path.join(BASE_DIR, 'recordings_cache')
SYSTEM_KEY = os.environ.get('CB_SYSTEM_KEY', 'default_system')

#class for storing timestamp of last video capture start and/or last snapshot to be used so no other video capture or snapshot is started before retrigger delay
class LastCaptureTime:
    def __init__(self):
        self.last_video_capture = None
        self.last_snapshot = None
        
TIME_UNITS = {
    'minutes': 60,
    'hours': 3600,
    'days': 86400
}

SUPPORTED_IMAGE_FILE_TYPES = ['png', 'jpg']

def clear_recordings(task_id, task_uuid='*', camera_id='*'):
    """Find and delete all directories/files in both temp and cache locations"""
    temp_pattern = os.path.join(TEMP_DIR, SYSTEM_KEY, camera_id, task_id, task_uuid)
    cache_pattern = os.path.join(RECORDINGS_CACHE_DIR, SYSTEM_KEY, task_id, "outbox", camera_id, "scheduled_recording", f"{task_uuid}_annotated")

    remove_files(temp_pattern)
    remove_files(cache_pattern)

def remove_files(pattern):
    """Remove files matching a pattern"""
    for path in glob.glob(pattern):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def ensure_dir_exists(directory):
    """Ensure a directory exists, creating it if necessary with better error handling"""
    try:
        os.makedirs(directory, exist_ok=True)
        #Test write permissions
        if not os.access(directory, os.W_OK):
            print(f"WARNING: Directory {directory} is not writable")
            #Try to use a fallback directory
            fallback = os.path.join(os.path.expanduser("~"), "iva_recordings")
            os.makedirs(fallback, exist_ok=True)
            print(f"Using fallback directory: {fallback}")
            return fallback
        return directory
    except PermissionError:
        print(f"ERROR: Permission denied when creating directory: {directory}")
        #Use fallback directory in user's home
        fallback = os.path.join(os.path.expanduser("~"), "iva_recordings")
        os.makedirs(fallback, exist_ok=True)
        print(f"Using fallback directory: {fallback}")
        return fallback
    except OSError as e:
        print(f"ERROR: Failed to create directory {directory}: {e}")
        #Create a safe directory name by removing problematic characters
        safe_dir = os.path.join(os.path.expanduser("~"), "iva_recordings")
        os.makedirs(safe_dir, exist_ok=True)
        print(f"Using fallback directory: {safe_dir}")
        return safe_dir

def get_video_paths(base_path):
    """Get all video paths in a directory structure sorted by creation time (oldest first)"""
    if not os.path.exists(base_path):
        return []
    video_files = []
    for root, _, files in os.walk(base_path):
        for file in files:
            if file.endswith(('.mp4', '.avi')):
                video_files.append(os.path.join(root, file))
    #Sort by creation time (using filename which contains the timestamp)
    return sorted(video_files, key=lambda x: timestamp_from_filename(os.path.basename(x)))

def clean_cache(base_path, max_videos=1):
    """Clean up the cache, keeping only the most recent videos up to max_videos"""
    videos = get_video_paths(base_path)
    #If we have more than max_videos, delete the oldest ones
    if len(videos) > max_videos:
        videos_to_delete = videos[:-max_videos]
        for video in videos_to_delete:
            os.remove(video)  
        #Clean up empty date directories
        for root, dirs, files in os.walk(base_path, topdown=False):
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                if not os.listdir(dir_path):
                    os.rmdir(dir_path)

def timestamp_from_filename(filename):
    """Extract timestamp from filename format YYYY-MM-DD_HH.MM.SS.ext"""
    basename = os.path.basename(filename)
    timestamp_str = os.path.splitext(basename)[0]
    try:
        dt = datetime.strptime(timestamp_str, DATETIME_FORMAT)
        return dt
    except ValueError:
        return None

def create_video_path(camera_id, task_id, task_uuid, root_path, file_type='mp4'):
    """Create a path for saving a video with current timestamp with better error handling"""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H.%M.%S')
    timestamp = f"{date_str}_{time_str}"
   
    path_base = root_path['path']
    path_id = root_path['id']
        
    if path_base.startswith('./'):
        path_base = path_base[2:]
    
    if not os.path.isabs(path_base):
        path_base = os.path.join(BASE_DIR, path_base)
            
    #Create path using object structure
    save_dir = os.path.join(path_base, SYSTEM_KEY, path_id, 'outbox', camera_id, task_id, task_uuid, date_str)
    
    try:
        save_dir = ensure_dir_exists(save_dir)
        return os.path.join(save_dir, f"{timestamp}.{file_type.lower()}")
    except Exception as e:
        #Last resort fallback
        print(f"ERROR creating path: {e}")
        fallback = os.path.join(os.path.expanduser("~"), "iva_recordings", f"{timestamp}.{file_type.lower()}")
        os.makedirs(os.path.dirname(fallback), exist_ok=True)
        return fallback

def add_to_shared_memory(task_uuid, frame):
        try:
            sm = shm.SharedMemory(name=f"{task_uuid}_frame")
            sm.buf[:frame.nbytes] = frame.tobytes()
        except FileNotFoundError:
            sm = shm.SharedMemory(create=True, size=np.prod(frame.shape) * np.dtype(np.uint8).itemsize, name=f"{task_uuid}_frame")
            sm.buf[:frame.nbytes] = frame.tobytes()
            
def combine_and_trim_video(video_paths, output_path, start_time, clip_length):
    """
    Combine multiple videos into one and trim to specified start time and length
    """
    if not video_paths:
        return False
       
    total_expected_seconds = 0
    videos_info = []
    fps_sum = 0
    
    for path in video_paths:
        if not os.path.exists(path):
            return False
            
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return False
            
        #Get key properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        
        fps_sum += fps
        
        videos_info.append({
            'path': path,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'width': width,
            'height': height
        })
        
        total_expected_seconds += duration
        cap.release()
    
    #Get properties from first video for output
    fps = fps_sum / len(videos_info)
    width = videos_info[0]['width']
    height = videos_info[0]['height']
   
    #Create temporary file for combined video
    import tempfile
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
   
    #Create output writer for temporary file
    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if temp_file.lower().endswith('.mp4') else 'XVID'))
    out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
   
    #Process each video for combination
    total_frames_written = 0
    for info in videos_info:
        cap = cv2.VideoCapture(info['path'])
        frames_from_this_video = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            frames_from_this_video += 1
            total_frames_written += 1
        
        cap.release()
   
    out.release()
   
    #Get video start time from the first filename
    video_start_time = timestamp_from_filename(os.path.basename(video_paths[0]))
    if not video_start_time:
        print(f"Could not determine start time from filename: {video_paths[0]}")
        os.unlink(temp_file)
        return None
       
    #Convert timestamps to seconds
    video_start_seconds = time.mktime(video_start_time.timetuple()) + video_start_time.microsecond / 1E6
    
    if isinstance(start_time, datetime):
        start_seconds = time.mktime(start_time.timetuple()) + start_time.microsecond / 1E6
    else:
        start_seconds = start_time
   
    #Calculate timing information
    offset_seconds = max(0, start_seconds - video_start_seconds)
   
    #Open the combined video
    cap = cv2.VideoCapture(temp_file)
    if not cap.isOpened():
        os.unlink(temp_file)
        return False
        
    #Check combined video length
    combined_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #Calculate frames to skip and frames to write
    start_frame = int(offset_seconds * fps)
    frames_to_write = int(clip_length * fps)
   
    #Create final output writer
    fourcc = cv2.VideoWriter_fourcc(*('mp4v' if output_path.endswith('.mp4') else 'XVID'))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
   
    #Skip to the start frame
    result = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if not result:
        #Manual seeking if direct positioning fails
        for _ in range(start_frame):
            cap.read()
   
    #Read and write frames
    frames_written = 0
    target_frames = min(frames_to_write, combined_frame_count - start_frame)
    
    while frames_written < target_frames:
        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: Ran out of frames at {frames_written}/{target_frames}")
            break
        
        out.write(frame)
        frames_written += 1
   
    #Release resources
    cap.release()
    out.release()
        
    #Delete the temporary file
    os.unlink(temp_file)
   
    print(f"Combined and trimmed video saved to {output_path}")
    return output_path

def setup_temp_dir(camera_id, task_id, task_uuid):
    """Set up the temporary directory for assembling videos"""
    temp_dir = ensure_dir_exists(os.path.join(TEMP_DIR, SYSTEM_KEY, camera_id, task_id, task_uuid))
    return temp_dir

def clear_directory(dir):
    """Clear temporary recordings directory"""
    if os.path.exists(dir):
        for file in os.listdir(dir):
            file_path = os.path.join(dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

def find_videos_for_time_range(videos, start_time, end_time, video_length):
    """Find videos that cover a specific time range"""
    relevant_videos = []
   
    for video in videos:
        video_time = timestamp_from_filename(video)
        if not video_time:
            continue
        
        video_end_time = video_time + timedelta(seconds=video_length)
       
        #Check if this video overlaps with our target range
        if (video_time <= end_time and video_end_time >= start_time):
            relevant_videos.append(video)
   
    return sorted(relevant_videos, key=lambda x: timestamp_from_filename(x))

def get_quality_perc(resolution):
    res = resolution.lower()
    if res == 'original':
        return 100
    elif res == 'lower':
        return 75
    return 50

def handle_snapshot_recording(camera_id, event, last_snapshot_time, retrigger_delay, root_path, file_type, task_id, task_uuid):
    current_time = time.time()
    saved_path = None
    
    if event and (current_time - last_snapshot_time >= retrigger_delay):
        updated_snapshot_time = current_time
        
        timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
        subfolder = timestamp.split('_')[0]
        
        system_key = os.environ.get('CB_SYSTEM_KEY', 'default_system')
        
        save_dir = f'{root_path["path"]}/{system_key}/{root_path["id"]}/outbox/{camera_id}/{task_id}/{task_uuid}/{subfolder}'
        ensure_dir_exists(save_dir)
        
        if file_type not in SUPPORTED_IMAGE_FILE_TYPES:
            file_type = 'png'
            
        saved_path = f'{save_dir}/{timestamp}.{file_type}'
        
        return updated_snapshot_time, saved_path
        
    return last_snapshot_time, None

def adjust_resolution(frame, quality):
    resolution = get_quality_perc(quality)
    if resolution == 100:
        return frame
    return cv2.resize(frame, (0, 0), fx=resolution/100, fy=resolution/100, interpolation=cv2.INTER_AREA)

def process_task_settings(task_settings):
    processed = {}
    
    processed['root_path'] = task_settings.get('root_path')
    
    processed['file_type'] = task_settings.get('file_type', '').lower()
    
    clip_length = int(task_settings.get('clip_length', 0))
    clip_length_units = task_settings.get('clip_length_units', 'seconds').lower()
    processed['clip_length'] = clip_length * TIME_UNITS.get(clip_length_units, 1)
    
    retrigger_delay = int(task_settings.get('retrigger_delay', 15))
    retrigger_delay_units = task_settings.get('retrigger_delay_units', 'seconds').lower()
    processed['retrigger_delay'] = retrigger_delay * TIME_UNITS.get(retrigger_delay_units, 1)
    
    processed['recording_lead_time'] = int(task_settings.get('recording_lead_time', 10) * TIME_UNITS.get(task_settings.get('recording_lead_time_units', 'seconds').lower(), 1))
    
    processed['resolution'] = task_settings.get('resolution', 'Lowest')
    
    processed['needs_video'] = processed['clip_length'] > 0
    processed['needs_snapshot'] = not processed['needs_video'] and processed['file_type'] in SUPPORTED_IMAGE_FILE_TYPES
    return processed

def trigger_scheduled_recording(camera_id, task_uuid, task_id, interval, adapter, quality, frame_shape):
    """Trigger scheduled_recording to create a continuous cache of clips"""
    #Create settings for scheduled_recording with root_path as object
    cache_dir = {
        "path": RECORDINGS_CACHE_DIR, 
        "id": task_id
    }
    
    lead_settings = {
        "root_path": cache_dir,
        "file_type": "mp4",
        "resolution": quality,
        "interval": interval,
        "units": "seconds",
        "clip_length": interval,
        "clip_length_units": "seconds",
        "start_time": "2020-01-01T00:00:00.000Z"
    }
   
    #Create message to send to scheduled_recording
    message = {
        "camera_id": camera_id,
        "uuid": f"{task_uuid}",
        "task_settings": lead_settings,
        "frame_shape": frame_shape,
        "publish_path": []  #No need to publish the output
    }
   
    #Send to scheduled_recording
    adapter.publish('task/scheduled_recording/input', json.dumps(message))

def setup_event_recording(camera_id, task_uuid, recording_lead_time, clip_length, adapter, task_id, quality, frame_shape):
    """Set up event recording for a camera"""
    #incoming task_uuid should end with '_annotated'
    cache_base_path = os.path.join(RECORDINGS_CACHE_DIR, SYSTEM_KEY, task_id, "outbox", camera_id, 'scheduled_recording', task_uuid)
   
    setup_temp_dir(camera_id, task_id, task_uuid.split('_annotated')[0])
   
    clean_cache(cache_base_path)
   
    interval = max(recording_lead_time, min(clip_length, 5))
    if recording_lead_time == 0:
        interval = min(max(clip_length//2, 5), 300)
        
    trigger_scheduled_recording(camera_id, task_uuid, task_id, interval, adapter, quality, frame_shape)
   
    return cache_base_path

def handle_event_recording(
    camera_id,
    event,
    last_event_time,
    recording_lead_time,
    clip_length,
    cache_base_path,
    root_path,
    file_type,
    task_id,
    task_uuid,
    task_output_data=None
):
    """Handle event recording logic with clear state transitions
   
    Args:
        camera_id: Camera identifier
        event: Boolean indicating if an event was detected
        last_event_time: Last time event was detected
        recording_lead_time: Seconds to record before event
        clip_length: Total clip length in seconds
        cache_base_path: Path to the cache directory
        temp_dir: Temporary directory for assembling videos
        root_path: Root path object with path and id properties
        file_type: Video file type (mp4 or avi)
        task_id: Task identifier
        task_output_data: Dictionary to update with output path
       
    Returns:
        updated_capture_time: Updated last_event_time value
    """
    current_time = time.time()
    
    #Create output path based on root_path object
    output_path = create_video_path(camera_id, task_id, task_uuid, root_path, file_type)
       
    #Path to temporary recordings for this camera/task
    temp_dir_path = os.path.join(TEMP_DIR, SYSTEM_KEY, camera_id, task_id, f"{task_uuid}")
    ensure_dir_exists(temp_dir_path)
   
    #Get existing videos in temp directory
    temp_videos = get_video_paths(temp_dir_path)
       
    #Post-event recording duration
    post_event_duration = clip_length - recording_lead_time
   
    #Determine our current state
    active_recording = len(temp_videos) > 0
    time_since_last_event = current_time - last_event_time
    
    event_datetime = datetime.fromtimestamp(last_event_time)
    expected_end_time = event_datetime + timedelta(seconds=clip_length-recording_lead_time)
    
    #STATE 1: New event detected, no active recording session, begin collecting cache footage
    if event and not active_recording:    
        #Update the last event time
        updated_capture_time = current_time
    
        #Clear any existing temporary recordings
        clear_directory(temp_dir_path)
        
        #Get all videos from cache
        cache_videos = get_video_paths(cache_base_path)
    
        #Copy all cache videos to temp directory
        for video in cache_videos:
            shutil.copy2(video, os.path.join(temp_dir_path, os.path.basename(video)))
        
        return updated_capture_time
   
    #STATE 2: Active recording session, continue to collect footage
    elif active_recording:       
        #Get all videos from cache and temp
        cache_videos = get_video_paths(cache_base_path)
        temp_videos = get_video_paths(temp_dir_path)       
        temp_filenames = [os.path.basename(v) for v in temp_videos]
        for video in cache_videos:
            video_name = os.path.basename(video)
            if video_name not in temp_filenames:
                shutil.copy2(video, os.path.join(temp_dir_path, video_name))
       
        #Get updated list of videos in temp
        temp_videos = get_video_paths(temp_dir_path)
       
        #Check if we have enough footage to cover the entire clip
        have_enough_footage = False
        if temp_videos:
            #Find the latest video timestamp
            latest_video = max(temp_videos, key=lambda x: timestamp_from_filename(os.path.basename(x)) or datetime.min)
            latest_timestamp = timestamp_from_filename(os.path.basename(latest_video))
           
            if latest_timestamp:
                #Assume each video is 5 seconds long
                latest_video_end = latest_timestamp + timedelta(seconds=5)

                #Check if we have enough footage
                have_enough_footage = latest_video_end >= expected_end_time
       
        #Only advance to STATE 3 if:
        #1. Enough time has passed since the event
        #2. We have enough footage to cover the whole clip
        #3. OR if it's been an excessive amount of time (safety timeout)
        excessive_wait = time_since_last_event > (post_event_duration + max(clip_length//2, 30))  #safety timeout
       
        if (time_since_last_event > post_event_duration and have_enough_footage) or excessive_wait:
           
            #Make sure we have some videos to work with
            if len(temp_videos) == 0:
                return last_event_time
                      
            #Start trimming recording_lead_time seconds before the event
            trim_start_time = last_event_time - recording_lead_time
           
            result = combine_and_trim_video(temp_videos, output_path, trim_start_time, clip_length)
           
            if result:               
                #Update output data
                if task_output_data is not None:
                    task_output_data["saved_path"] = output_path
               
                #Clean up temp directory
                clear_directory(temp_dir_path)
               
                #Reset last_event_time
                return 0
            else:
                print("Failed to generate video")
                #Reset last_event_time
                return 0
       
        return last_event_time
   
    #Default case
    return last_event_time
