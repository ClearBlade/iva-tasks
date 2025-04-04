import cv2
from shapely.geometry import LineString, Polygon, Point
from collections import defaultdict
import numpy as np

DIRECTION_A_TO_B = "A_TO_B"
DIRECTION_B_TO_A = "B_TO_A"

UI_SCALE = [360, 640]

def line_intersects_box(line, box):
    line = LineString(line)
    box = Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
    return line.intersects(box)

def is_point_on_side_B(line, point):
    x1, y1 = line[0]
    x2, y2 = line[1]
    
    if isinstance(point, Point):
        px, py = point.x, point.y
    else:
        px, py = point
    
    if x2 - x1 == 0:
        return px < x1
    
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    y_on_line = m * px + b
    
    if m > 0:
        return py < y_on_line
    elif m < 0:
        return py > y_on_line
    else:
        return py < y1

def determine_direction(line, prev_point, curr_point):
    prev_on_B = is_point_on_side_B(line, prev_point)
    curr_on_B = is_point_on_side_B(line, curr_point)
    if prev_on_B and not curr_on_B:
        return DIRECTION_B_TO_A
    elif not prev_on_B and curr_on_B:
        return DIRECTION_A_TO_B
    else:
        return None

def get_important_vertices(line, box):
    x1, y1, x2, y2 = box
    if (line[0][0] == line[1][0]) or ((line[1][1] - line[0][1]) / (line[1][0] - line[0][0]) < 0):
        return [(x1, y2), (x2, y1)] #top-left and bottom-right
    else:
        return [(x1, y1), (x2, y2)] #bottom-left and top-right

def are_objects_close(obj1, obj2, threshold=150):
    #Calculate centroids
    centroid1 = ((obj1['box_points'][0] + obj1['box_points'][2]) / 2,
                 (obj1['box_points'][1] + obj1['box_points'][3]) / 2)
    centroid2 = ((obj2['box_points'][0] + obj2['box_points'][2]) / 2,
                 (obj2['box_points'][1] + obj2['box_points'][3]) / 2)
    
    #Calculate horizontal and vertical distances
    hDistance = abs(centroid1[0] - centroid2[0])
    vDistance = abs(centroid1[1] - centroid2[1])
    if hDistance < threshold and vDistance < threshold:
        return True
    return False

def calculate_side_points(start_point, end_point, distance=33):
    midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)    
    dx = end_point[0] - start_point[0]
    dy = end_point[1] - start_point[1]

    #Normalize the direction vector
    length = (dx**2 + dy**2)**0.5
    if length != 0:
        dx, dy = dx / length, dy / length

    #Calculate the perpendicular vector
    perpendicular_x, perpendicular_y = -dy, dx

    #Calculate points A and B
    point_1 = (midpoint[0] + perpendicular_x * distance,
            midpoint[1] + perpendicular_y * distance)
    point_2 = (midpoint[0] - perpendicular_x * distance,
            midpoint[1] - perpendicular_y * distance)

    return point_1, point_2

def get_text_vertical_offset(start_point, end_point): #take the start and end points of the line
    dx, dy = end_point[0] - start_point[0], end_point[1] - start_point[1]
    angle = np.arctan2(dy, dx)
    #hoffset = int(20 * abs(np.sin(angle)))
    voffset = int(50 * abs(np.cos(angle)))
    return voffset

def detect_crossing(line, current_box, previous_box):
    prev_center = ((previous_box[0] + previous_box[2]) / 2,
                   (previous_box[1] + previous_box[3]) / 2)
    curr_center = ((current_box[0] + current_box[2]) / 2,
                     (current_box[1] + current_box[3]) / 2)
    direction = determine_direction(line, prev_center, curr_center)
    if direction: #past and current centroids are on different sides of the line, potential crossing
        current_vertices = get_important_vertices(line, current_box)
        previous_vertices = get_important_vertices(line, previous_box)
        
        #Create a rectangle between past and current frames
        rectangle = Polygon([current_vertices[0], current_vertices[1], 
                            previous_vertices[1], previous_vertices[0]])
        
        #Check if line intersects the rectangle
        if line_intersects_box(line, rectangle.bounds):
            
            return direction
    
    return None

#Returns the average color in RGB format
def get_average_color(image, box_coordinates):
    x, y, w, h = box_coordinates
    roi = image[y:y+h, x:x+w]
    avg_color = np.mean(roi, axis=(0, 1))
    return (avg_color[2], avg_color[1], avg_color[0])

#expects color in RGB format
def calculate_luminance(color):
    return 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

def choose_colors(image, box_coordinates):
    avg_color = get_average_color(image, box_coordinates)
    luminance = calculate_luminance(avg_color)
    colors = {'line_color': None, 'text_color': None}
    
    #In BGR format
    if luminance < 128:
        colors['line_color'] = (175, 235, 39)  #ClearBlade Green/Turquoise dark backgrounds
    else:
        colors['line_color'] = (102, 77, 4)  #Navy for light backgrounds  
    if luminance < 57:
        colors['text_color'] = (255, 255, 255) #White text for dark backgrounds
    else:
        colors['text_color'] = (0, 0, 0) #Black text for light backgrounds
    return colors

def rescale_line(line, target_shape, current_shape):
    x1, y1, x2, y2 = line
    #Calculate width and height scaling factors directly
    width_scale = target_shape[1] / current_shape[1]
    height_scale = target_shape[0] / current_shape[0]
    #Apply scaling to each coordinate
    scaled_x1 = int(x1 * width_scale)
    scaled_y1 = int(y1 * height_scale)
    scaled_x2 = int(x2 * width_scale)
    scaled_y2 = int(y2 * height_scale)
    return [scaled_x1, scaled_y1, scaled_x2, scaled_y2]

class CameraTracker:
    def __init__(self, initial_frame, frame_shape, line):
        self.previous_frame = {}
        self.current_frame = {}
        self.line = line
        #Create region of interest around the line to choose the best colors for the line and text
        x1, y1, x2, y2 = line[0][0], line[0][1], line[1][0], line[1][1]
        if x1 == x2:
            if x2 < frame_shape[1]:
                x2 += 1
            else:
                x1 -= 1
        if y1 == y2:
            if y2 < frame_shape[0]:
                y2 += 1
            else:
                y1 -= 1
        self.colors = choose_colors(initial_frame, (x1, y1, x2, y2))        

    def update(self, box_data, line):        
        self.previous_frame = self.current_frame
        self.current_frame = {label: {'box_points': [box[0], box[1], box[2], box[3]], 'cross': None} for label, box in box_data.items()}
        self.line = line

    def process_crossings(self, objects_to_detect, direction=None):
        #Relabel objects if necessary
        for classification in objects_to_detect:
            past_objects = [label for label in self.previous_frame if classification in label]
            current_objects = [label for label in self.current_frame if classification in label]
            
            if len(past_objects) == 1 and len(current_objects) == 1 and past_objects[0] != current_objects[0]:
                if are_objects_close(self.previous_frame[past_objects[0]], self.current_frame[current_objects[0]]):
                    self.previous_frame[current_objects[0]] = self.previous_frame.pop(past_objects[0])
        results = defaultdict(list)
        for label, current_obj in self.current_frame.items():
            previous_obj = self.previous_frame.get(label)
            if previous_obj:
                crossing = detect_crossing(self.line, current_obj['box_points'], previous_obj['box_points'])
                self.current_frame[label]['cross'] = crossing
                if crossing:
                    if direction is None or crossing == direction:
                        classification = ''.join([c for c in label if not c.isdigit()])
                        results[classification].append(crossing)
       
        for label in list(self.previous_frame.keys()):
            if label not in self.current_frame:
                self.previous_frame.pop(label)
       
        return results

    def draw_line(self, image):
        if self.line is not None:
            start_point = (int(self.line[0][0]), int(self.line[0][1]))
            end_point = (int(self.line[1][0]), int(self.line[1][1]))
            thickness = 3
            image = cv2.line(image, start_point, end_point, self.colors['line_color'], thickness)
            point_1, point_2 = calculate_side_points(start_point, end_point)
            if is_point_on_side_B(self.line, point_1):
                A, B = point_2, point_1
            else:
                A, B = point_1, point_2 
            A = (int(A[0]), int(A[1]))
            B = (int(B[0]), int(B[1]))
            v_offset = get_text_vertical_offset(start_point, end_point)
            #The text needs to be centered on A and B
            #so adjust the coordinates accordingly
            if A[1] < B[1]:
                A = (A[0] - 20, A[1])
                B = (B[0] - 20, B[1] + v_offset)
            else:
                A = (A[0] - 20, A[1] + v_offset)
                B = (B[0] - 20, B[1])
            image = cv2.putText(image, 'A', A, cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors['text_color'], 5, lineType = cv2.LINE_AA)
            image = cv2.putText(image, 'B', B, cv2.FONT_HERSHEY_SIMPLEX, 2, self.colors['text_color'], 5, lineType = cv2.LINE_AA)
        return image
    
if __name__ == '__main__':
    import os
    import time
    import sys
    import numpy as np
    import cv2
    from collections import deque
    
    #Add parent directory to path to import object_detection
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    
    #Save current working directory
    original_dir = os.getcwd()
    
    try:
        #Change directory to object_detection folder so it can find its assets
        os.chdir(os.path.join(parent_dir, 'object_detection'))
        
        #Import after changing directory
        from object_detection.object_detection import detect_objects
        
        #Create a test environment
        os.environ['CB_SYSTEM_KEY'] = "test_system_key"
        
        #Test video path - provide an absolute path to your test video
        video_path = os.path.abspath('/your/path/to/test_video.mp4')
        
        #Define your test line
        test_line = [[740, 0], [741, 720]]
        
        #Open the test video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video file: {video_path}")
            exit(1)
        
        #Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        spf = 1.0 / fps  #seconds per frame in the original video
        print(f"Video FPS: {fps}, Seconds per frame: {spf:.6f}")
        
        #Initialize with a default frame skip
        frame_skip = 1
        
        #Read the first frame for initialization
        ret, initial_frame = cap.read()
        if not ret:
            print("Failed to read the first frame")
            exit(1)
        
        #Get frame dimensions
        height, width = initial_frame.shape[:2]
        frame_shape = (width, height)
        
        #Initialize the tracker
        tracker = CameraTracker(initial_frame, frame_shape, test_line)
        
        #Create object detection settings
        object_detection_settings = {
            "objects_to_detect": {
                "person": {
                    "enable_tracking": True,
                    "enable_blur": False,
                    "show_boxes": True,
                    "show_labels": True,
                    "confidence_threshold": 0.35
                }
            }
        }
        
        #Create a window to display the frames
        cv2.namedWindow('Line Crossing Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Line Crossing Test', width // 2, height // 2)
        
        frame_count = 0
        processed_count = 0
        
        #For calculating average processing time
        process_times = deque(maxlen=10)
        calibration_complete = False
        
        print("Testing line crossing detection with real object detection...")
        print("Calibrating frame skip rate with first 10 frames...")
        
        #Variable for pause timing
        pause_until = 0
        last_frame = None
        
        while True:
            #Check if we're in a pause state (for crossing detected)
            current_time = time.time()
            if current_time < pause_until:
                #During pause, just refresh the display and wait
                if last_frame is not None:
                    cv2.imshow('Line Crossing Test', last_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                time.sleep(0.01)  #Small sleep to avoid CPU hogging
                continue
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            #Process only selected frames based on frame_skip
            #During calibration, process all frames
            if not calibration_complete or frame_count % frame_skip == 0:
                process_start = time.time()
                
                #Use object detection to get bounding boxes
                try:
                    annotated_frame, bboxes, objects_detected, total_objects = detect_objects(
                        "test_camera", object_detection_settings, frame.copy(), frame_shape
                    )
                    
                    #Format bounding boxes for the tracker
                    box_data = {}
                    for label, bbox in bboxes.items():
                        if isinstance(bbox[0], list):  #Handle multiple detections of same class
                            for i, box in enumerate(bbox):
                                box_data[f"{label}_{i}"] = box
                        else:
                            box_data[label] = bbox
                    
                    #Update tracker with detected boxes
                    tracker.update(box_data, test_line)
                    
                    #Process crossings
                    results = tracker.process_crossings(['person'], None)
                    
                    #Draw the line on the frame
                    annotated_frame = tracker.draw_line(annotated_frame)
                    
                    #Check if a crossing was detected
                    if results and 'person' in results and results['person']:
                        crossing_direction = results['person'][0]
                        direction_text = "A to B" if crossing_direction == DIRECTION_A_TO_B else "B to A"
                        action_text = "entered" if crossing_direction == DIRECTION_A_TO_B else "exited"
                        
                        print(f"Person {action_text}! Direction: {direction_text}")
                        
                        #Add text to the frame
                        cv2.putText(
                            annotated_frame, 
                            f"PERSON {action_text.upper()}!", 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255), 
                            2, 
                            cv2.LINE_AA
                        )
                        
                        #Set pause time to freeze the frame
                        pause_until = time.time() + 0.5  #Pause for 0.5 seconds
                    
                    #Save the current frame for displaying during pause
                    last_frame = annotated_frame.copy()
                    
                    #Display the frame
                    cv2.imshow('Line Crossing Test', annotated_frame)
                    
                    #Record processing time
                    process_time = time.time() - process_start
                    process_times.append(process_time)
                    processed_count += 1
                    
                    #After 10 frames, calculate the optimal frame skip to maintain real-time playback
                    if not calibration_complete and len(process_times) == 10:
                        avg_process_time = sum(process_times) / len(process_times)
                        print(f"Average processing time per frame: {avg_process_time:.6f} seconds")
                        
                        #Calculate how many frames to skip to maintain real-time playback
                        #If process_time > spf, we need to skip frames
                        if avg_process_time > spf:
                            frame_skip = max(1, int(avg_process_time / spf))
                            print(f"Setting frame skip to {frame_skip} to maintain real-time playback")
                        else:
                            frame_skip = 1
                            print("Processing every frame (no skipping needed)")
                            
                        calibration_complete = True
                
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    process_times.append(0.1)  #Use a default value for errors
                    if last_frame is None:
                        last_frame = frame.copy()
            
            #Add a small delay to make display smoother
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            #Status update every 100 processed frames
            if processed_count % 100 == 0 and processed_count > 0:
                print(f"Processed {processed_count} frames (read {frame_count} frames)")
            
            #Break after a reasonable number of frames
            if frame_count >= 10000:
                print("Test completed after maximum number of frames")
                break
        
        #Clean up
        cap.release()
        cv2.destroyAllWindows()
        print(f"Test completed. Processed {processed_count} out of {frame_count} frames.")
        if len(process_times) > 0:
            print(f"Final average processing time: {sum(process_times)/len(process_times):.6f} seconds")
    
    except Exception as e:
        print(f"Test error: {e}")
    
    finally:
        #Change back to original directory
        os.chdir(original_dir)
