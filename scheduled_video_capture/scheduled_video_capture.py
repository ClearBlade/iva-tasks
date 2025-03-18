import os
from datetime import datetime, timezone
import cv2
import dateutil.parser as parser
import numpy as np

time_units = {
    "Minutes": 60,
    "Hours": 3600,
    "Days": 86400
}

#Class to manage video capture sessions for multiple cameras
class VideoCaptureSessions:
    def __init__(self):
        self.sessions = {}  #Format: {camera_id: {task_uuid: {first_timestamp, last_timestamp, writer, frames}}}
        self.idling = {}  #Format: {camera_id: {task_uuid: {last_interval_position}}}
    
    def should_start_capture(self, camera_id, task_uuid, timestamp, start_time, interval):
        """Determine if a new capture session should start based on timestamps and interval"""
        #Parse timestamps
        timestamp_dt = parser.parse(timestamp).astimezone()
        start_time_dt = parser.parse(start_time).astimezone()
        
        #Check if we're past the start time
        if timestamp_dt < start_time_dt:
            return False

        #Calculate time since start
        time_since_start = (timestamp_dt - start_time_dt).total_seconds()

        #Calculate the current interval position
        current_interval_position = time_since_start % interval
        
        #If current interval position is less than the last interval position, we should start
        should_start = current_interval_position <= self.idling.get(camera_id, {}).get(task_uuid, {}).get('last_interval_position', 0)
        if should_start:
            self.idling.get(camera_id, {}).get(task_uuid, {})['last_interval_position'] = 0
        else:
            self.idling.get(camera_id, {}).get(task_uuid, {})['last_interval_position'] = current_interval_position
        return should_start
        
    
    def start_session(self, camera_id, task_uuid, timestamp, frame_shape):
        """Start a new video capture session"""
        if camera_id not in self.sessions:
            self.sessions[camera_id] = {}
        self.sessions[camera_id][task_uuid] = {
            'first_timestamp': timestamp,
            'last_timestamp': timestamp,
            'frames': [],
            'frame_durations': [],  #Store duration for each frame
            'frame_shape': frame_shape
        }
        
    def add_frame(self, camera_id, task_uuid, frame, timestamp):
        """Add a frame to an existing session"""
        if camera_id in self.sessions and task_uuid in self.sessions[camera_id]:
            session = self.sessions[camera_id][task_uuid]
            
            if frame is None:
                print(f"ERROR: Null frame received for camera {camera_id}, task {task_uuid} at {timestamp}")
                return
                
            if len(frame.shape) < 2 or frame.size == 0:
                print(f"ERROR: Invalid frame shape {frame.shape} for camera {camera_id}, task {task_uuid} at {timestamp}")
                return
            
            #Calculate duration of previous frame if there is one
            if session['frames']:
                last_timestamp_dt = parser.parse(session['last_timestamp'])
                current_timestamp_dt = parser.parse(timestamp)
                duration = (current_timestamp_dt - last_timestamp_dt).total_seconds()
                session['frame_durations'].append(max(duration, 0.033))  #At least 1/30 second
                        
            #Add the new frame - create a deep copy to ensure we don't have reference issues
            frame_copy = frame.copy()
            session['frames'].append(frame_copy)
            session['last_timestamp'] = timestamp

    def should_end_session(self, camera_id, task_uuid, timestamp, duration):
        """Check if a session should end based on duration"""
        if camera_id in self.sessions and task_uuid in self.sessions[camera_id]:
            session = self.sessions[camera_id][task_uuid]
            first_timestamp_dt = parser.parse(session['first_timestamp'])
            timestamp_dt = parser.parse(timestamp)
            
            #Check if duration exceeded
            elapsed = (timestamp_dt - first_timestamp_dt).total_seconds()
            return elapsed >= duration
        
        return False
    
    def end_session(self, camera_id, task_uuid, root_path, resolution, file_type, task_id, duration):
        """End a session and save the video"""
        if camera_id in self.sessions and task_uuid in self.sessions[camera_id]:
            session = self.sessions[camera_id][task_uuid]
        
            #Get timestamps for naming
            first_timestamp_dt = parser.parse(session['first_timestamp'])
            first_timestamp_local = first_timestamp_dt.astimezone()
        
            #Format date and time for folder structure
            date_str = first_timestamp_local.strftime("%Y-%m-%d")
            vid_name = first_timestamp_local.strftime("%Y-%m-%d_%H.%M.%S")
        
            #Create save path
            system_key = os.environ.get('CB_SYSTEM_KEY', 'default_system')
            save_path = os.path.join(root_path, system_key, camera_id, task_id, date_str)
            os.makedirs(save_path, exist_ok=True)
        
            #Full path with filename
            file_path = os.path.join(save_path, f"{vid_name}.{file_type.lower()}")
        
            #Get frame info
            frames = session['frames']
                        
            if not frames:
                print(f"ERROR: No frames to save for session {task_uuid}")
                del self.sessions[camera_id][task_uuid]
                return ""
            
            #Verify frames are valid before proceeding
            valid_frames = []
            invalid_count = 0
            for i, frame in enumerate(frames):
                if frame is not None and frame.size > 0 and len(frame.shape) >= 2:
                    valid_frames.append(frame)
                else:
                    invalid_count += 1
                    print(f"ERROR: Frame {i} is invalid in session {task_uuid}")
                    
            if invalid_count > 0:
                print(f"Found {invalid_count} invalid frames out of {len(frames)} total")
                
            if not valid_frames:
                print(f"ERROR: No valid frames to save for session {task_uuid}")
                del self.sessions[camera_id][task_uuid]
                return ""
                
            frames = valid_frames
            
            #Get video properties
            height, width = frames[0].shape[:2]
        
            #Apply resolution scaling
            scale_factor = get_quality_perc(resolution) / 100
            if scale_factor != 1:
                scaled_width = int(width * scale_factor)
                scaled_height = int(height * scale_factor)
                for i in range(len(frames)):
                    frames[i] = cv2.resize(frames[i], (scaled_width, scaled_height),
                                        interpolation=cv2.INTER_AREA)
                width, height = scaled_width, scaled_height
        
            #Calculate FPS based on actual frame count and duration
            fps = len(frames) / duration
            
            #Ensure FPS between 1 and 60
            fps = max(min(fps, 60.0), 1.0)
        
            #Choose codec based on file type
            if file_type.lower() == 'avi':
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
            else:
                #MP4 codec
                try:
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')  #H.264 codec
                except:
                    try:
                        print("WARNING: avc1 codec not available, falling back to H264")
                        fourcc = cv2.VideoWriter_fourcc(*'H264')
                    except:
                        print("WARNING: H264 codec not available, falling back to MP4V")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #Fallback
        
            #Create VideoWriter
            writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
            
            if not writer.isOpened():
                print(f"ERROR: Failed to open VideoWriter for {file_path}")
                del self.sessions[camera_id][task_uuid]
                return ""
        
            #Write all frames
            for i, frame in enumerate(frames):
                try:
                    writer.write(frame)
                except Exception as e:
                    print(f"ERROR: Failed to write frame {i}: {str(e)}")
        
            #Release resources
            writer.release()
        
            #Clean up
            result_path = file_path
            del self.sessions[camera_id][task_uuid]
        
            return result_path

        return ""

def get_quality_perc(resolution):
    if resolution == 'Original':
        return 100
    elif resolution == 'Lower':
        return 75
    return 50

#Create a global session manager
capture_sessions = VideoCaptureSessions()

def save_frame(frame, camera_id, task_uuid, task_settings, task_id):    
    #Make a deep copy of the frame to avoid reference issues
    if frame is not None:
        try:
            frame = frame.copy()            
        except Exception as e:
            print(f"ERROR: Failed to copy frame: {str(e)}")
            frame = None
    
    if frame is None or frame.size == 0 or len(frame.shape) < 2:
        print(f"ERROR: Invalid frame received for {camera_id}, task {task_uuid}")
        return ""
        
    root_path = task_settings.get("root_path", "./assets/saved_videos")
    file_type = task_settings.get("file_type", "mp4")
    resolution = task_settings.get("resolution", "Lowest")
    interval = int(task_settings.get("interval", 3600))
    interval_units = task_settings.get("interval_units", "Seconds")
    if interval_units != "Seconds":
        interval *= time_units.get(interval_units, 1)
    duration = int(task_settings.get("clip_length", interval))
    duration_units = task_settings.get("clip_length_units", "Seconds")
    if duration_units != "Seconds":
        duration *= time_units.get(duration_units, 1)
    start_time = task_settings.get("start_time", datetime.now(timezone.utc).isoformat())
    timestamp = datetime.now(timezone.utc).isoformat()
   
    #Get frame shape for initialization
    frame_shape = frame.shape
   
    saved_file_path = ""  #Initialize return value
    
    #If idling is not set up, initialize it
    if camera_id not in capture_sessions.idling:
        capture_sessions.idling[camera_id] = {}
    if task_uuid not in capture_sessions.idling[camera_id]:
        capture_sessions.idling[camera_id][task_uuid] = { 'last_interval_position': 0 }
    
    should_start = capture_sessions.should_start_capture(camera_id, task_uuid, timestamp, start_time, interval)

    #Check if we have an active session and it should end
    active_session = camera_id in capture_sessions.sessions and task_uuid in capture_sessions.sessions[camera_id]
    should_end = active_session and capture_sessions.should_end_session(camera_id, task_uuid, timestamp, duration)
    
    if should_end or should_start:
        if active_session:
            #End the session and save video
            saved_file_path = capture_sessions.end_session(camera_id, task_uuid, root_path, resolution, file_type, task_id, duration)
   
    #Then check if we should start a new session (regardless of whether we just ended one)
    if (camera_id not in capture_sessions.sessions or
        task_uuid not in capture_sessions.sessions[camera_id]) and \
       should_start:
        #Start a new session
        capture_sessions.start_session(camera_id, task_uuid, timestamp, frame_shape)
   
    #If we have an active session now, add the frame
    if camera_id in capture_sessions.sessions and task_uuid in capture_sessions.sessions[camera_id]:
        capture_sessions.add_frame(camera_id, task_uuid, frame, timestamp)
   
    return saved_file_path

if __name__ == '__main__':
    #Test code here if needed
    pass