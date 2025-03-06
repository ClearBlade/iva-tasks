import os
from datetime import datetime, timezone
import cv2
import dateutil.parser as parser

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
            
            #Calculate duration of previous frame if there is one
            if session['frames']:
                last_timestamp_dt = parser.parse(session['last_timestamp'])
                current_timestamp_dt = parser.parse(timestamp)
                duration = (current_timestamp_dt - last_timestamp_dt).total_seconds()
                session['frame_durations'].append(max(duration, 0.033))  #At least 1/30 second
            
            #Add the new frame
            session['frames'].append(frame.copy())
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
                #No frames to save
                del self.sessions[camera_id][task_uuid]
                return ""
            
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
            if file_type.lower() == 'mp4':
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            else:  #AVI
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
        
            #Create VideoWriter
            writer = cv2.VideoWriter(file_path, fourcc, fps, (width, height))
        
            #Write all frames
            for frame in frames:
                writer.write(frame)
        
            #Release resources
            writer.release()
        
            #Clean up
            result_path = file_path
            del self.sessions[camera_id][task_uuid]
        
            return result_path
    
        return ""

def get_quality_perc(resolution):
    if resolution == 'High':
        return 100
    elif resolution == 'Medium':
        return 75
    return 50

#Create a global session manager
capture_sessions = VideoCaptureSessions()

def save_frame(frame, camera_id, task_uuid, task_settings, task_id):
    root_path = task_settings.get("root_path", "./assets/saved_videos")
    file_type = task_settings.get("file_type", "MP4")
    resolution = task_settings.get("resolution", "Low")
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
    if (camera_id in capture_sessions.sessions and
        task_uuid in capture_sessions.sessions[camera_id] and
        capture_sessions.should_end_session(camera_id, task_uuid, timestamp, duration)) or should_start:
       
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