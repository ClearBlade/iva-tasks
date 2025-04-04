import cv2
import numpy as np
from ultralytics import YOLO

models = {}
id_trackers = {"blur": {}}
colors = {}
supported_blur_classes = ["person"]

class TrackingInfo:
    def __init__(self, tracker_id, bbox, yolo_id=None):
        self.tracker_id = tracker_id
        self.bbox = bbox
        self.yolo_id = yolo_id
        
class IDTracker:
    def __init__(self, classifications, distance_threshold=500):
        self.classifications = classifications
        self.distance_threshold = distance_threshold
        self.tracking_info = {cls: {} for cls in classifications}
        self.untracked_objects = {cls: {} for cls in classifications}

    def update(self, detections):
        current_yolo_ids = {cls: set() for cls in self.classifications}
        untracked_detections = []
        new_tracking_info = {cls: {} for cls in self.classifications}
        assigned_tracker_ids = {cls: set() for cls in self.classifications}
        
        #First, handle all tracked objects
        for cls, yolo_id, bbox, _ in detections:
            if yolo_id >= 0:
                current_yolo_ids[cls].add(yolo_id)
                if yolo_id in self.tracking_info[cls]:
                    tracker_id = self.tracking_info[cls][yolo_id].tracker_id
                else:
                    tracker_id = self.handle_new_tracked_object(cls, bbox, yolo_id, assigned_tracker_ids)            
                if tracker_id in assigned_tracker_ids[cls]:
                    tracker_id = self.get_next_available_id(cls, assigned_tracker_ids[cls])            
                assigned_tracker_ids[cls].add(tracker_id)
                new_tracking_info[cls][yolo_id] = TrackingInfo(tracker_id, bbox, yolo_id)
            else:
                untracked_detections.append((cls, yolo_id, bbox))
    
        #Handle untracked objects
        self.update_pretracking(untracked_detections, new_tracking_info, assigned_tracker_ids)
    
        #Update tracking_info
        self.tracking_info = new_tracking_info
        return self.tracking_info

    def update_pretracking(self, untracked_detections, new_tracking_info, assigned_tracker_ids):
        for cls, yolo_id, bbox in untracked_detections:
            matched = False
            for tracker_id, info in list(self.untracked_objects[cls].items()):
                distance = get_distance_between_objects(bbox, info.bbox)
                if distance <= self.distance_threshold:
                    new_tracking_info[cls][yolo_id] = TrackingInfo(tracker_id, bbox, yolo_id)
                    matched = True
                    break
            if not matched:
                tracker_id = self.get_next_available_id(cls, assigned_tracker_ids[cls])
                assigned_tracker_ids[cls].add(tracker_id)
                new_tracking_info[cls][yolo_id] = TrackingInfo(tracker_id, bbox, yolo_id)
        #Update untracked_objects with new untracked detections
        self.untracked_objects = {cls: {} for cls in self.classifications}
        for cls, yolo_id in new_tracking_info.items():
            for info in new_tracking_info[cls].values():
                self.untracked_objects[cls][info.tracker_id] = info

    def handle_new_tracked_object(self, cls, bbox, yolo_id, assigned_tracker_ids):
        for tracker_id, info in self.untracked_objects[cls].items():
            distance = get_distance_between_objects(bbox, info.bbox)
            if distance <= self.distance_threshold:
                return tracker_id
        tracker_id = self.get_next_available_id(cls, assigned_tracker_ids[cls])
        self.tracking_info[cls][tracker_id] = TrackingInfo(tracker_id, bbox, yolo_id)
        return tracker_id

    def get_next_available_id(self, cls, additional_used_ids=None):
        used_ids = set(info.tracker_id for info in self.tracking_info[cls].values())
        used_ids.update(info.tracker_id for info in self.untracked_objects[cls].values())
    
        #Add any additional used IDs from the current frame
        if additional_used_ids:
            used_ids.update(additional_used_ids)
    
        tracker_id = 1
        while tracker_id in used_ids:
            tracker_id += 1
        return tracker_id

    def get_tracker_id(self, cls, yolo_id):
        if yolo_id in self.tracking_info[cls]:
            return self.tracking_info[cls][yolo_id].tracker_id
        else:
            for tracker_id, info in self.untracked_objects[cls].items():
                if info.yolo_id == yolo_id:
                    return tracker_id

def get_distance_between_objects(obj1, obj2):
    #Calculate centroids
    if obj1 is None or obj2 is None:
        return 99999.0
    centroid1 = ((obj1[0] + obj1[2]) / 2, (obj1[1] + obj1[3]) / 2)
    centroid2 = ((obj2[0] + obj2[2]) / 2, (obj2[1] + obj2[3]) / 2)
    #Calculate horizontal and vertical distances
    hDistance = abs(centroid1[0] - centroid2[0])
    vDistance = abs(centroid1[1] - centroid2[1])
    return (hDistance**2 + vDistance**2)**0.5

def get_model_and_tracker(camera_id, task_uuid, class_assignments, enable_tracking):
    #if at least one object has enable_tracking set to True, then all objects will be tracked, we will only display the tracking data for those with it enabled
    
    if enable_tracking:
        if camera_id not in models:
            models[camera_id] = {}
            id_trackers[camera_id] = {}
        if task_uuid not in models[camera_id]:
            models[camera_id][task_uuid] = YOLO("assets/yolo12s.onnx", task='detect')
            id_trackers[camera_id][task_uuid] = IDTracker(class_assignments.keys())
        return models[camera_id][task_uuid], id_trackers[camera_id][task_uuid]
    else:
        if camera_id not in models:
            models[camera_id] = {}
        if task_uuid not in models[camera_id]:
            models[camera_id][task_uuid] = YOLO("assets/yolo12s.onnx", task='detect')
        return models[camera_id][task_uuid], None

def get_colors(camera_id, task_uuid, class_assignments, frame, frame_shape):
    if camera_id not in colors:
        colors[camera_id] = {}
    if task_uuid not in colors[camera_id]:
        box_coords = (0, 0, frame_shape[1], frame_shape[0])
        colors[camera_id][task_uuid] = choose_colors(frame, box_coords, class_assignments)
    return colors[camera_id][task_uuid]

def get_average_color(image, box_coordinates):
    x, y, w, h = box_coordinates
    roi = image[y:y+h, x:x+w]
    avg_color = np.mean(roi, axis=(0, 1))
    return (avg_color[2], avg_color[1], avg_color[0])

def calculate_luminance(color):
    return 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]

def hsv_to_rgb(h, s, v):
    h = h / 60
    c = v * s
    x = c * (1 - abs(h % 2 - 1))
    m = v - c
   
    if h < 1:
        r, g, b = c, x, 0
    elif h < 2:
        r, g, b = x, c, 0
    elif h < 3:
        r, g, b = 0, c, x
    elif h < 4:
        r, g, b = 0, x, c
    elif h < 5:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
   
    return (r + m, g + m, b + m)

def choose_colors(frame, box_coords, classes):
    avg_color = get_average_color(frame, box_coords)
    luminance = calculate_luminance(avg_color)
    hsv_color = cv2.cvtColor(np.uint8([[avg_color]]), cv2.COLOR_BGR2HSV)[0][0]
    base_hue = hsv_color[0] / 2
    all_colors = {}
    hue_step = 360 // 20  #Generate a palette
   
    for i in range(20):
        hue = (base_hue + i * hue_step) % 360       
        if luminance > 128:
            saturation = 0.75
            value = 0.3
        elif luminance > 60:
            saturation = 0.99
            value = 0.8
        else:
            saturation = 0.8
            value = 1.0
        rgb_color = hsv_to_rgb(hue, saturation, value)
        bgr_color = (int(rgb_color[2] * 255), int(rgb_color[1] * 255), int(rgb_color[0] * 255))
        all_colors[i] = bgr_color
    #Select color for each class
    colorPalette = {}
    predefined_indices = [1,7,15,5,11]  #Predefined palette members to guarantee hue variance for first 5 classes
    available_colors = list(range(10))
    for i, class_name in enumerate(classes):
        if i < 5:
            color_key = predefined_indices[i]
        else:
            color_key = np.random.choice(available_colors)
        available_colors.remove(color_key)
        colorPalette[class_name] = all_colors[color_key]
    return colorPalette

def read_class_names(class_file_name): #loads class names from file
    names = {}
    with open(class_file_name, "r") as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip("\n")
    return names

def get_class_assignments(objects): #returns a dictionary of class assignments
    names = read_class_names("assets/coco.names")
    class_assignment = {}
    for key, value in names.items():
        if value in objects:
            class_assignment[value] = key  #Use string label as key, numeric ID as value
    return class_assignment

def draw_annotation(frame, overlay, label, conf, bbox, colorPalette, object_settings, cls):
    x1, y1, x2, y2 = bbox
    if object_settings.get("show_boxes", True):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorPalette[cls], 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colorPalette[cls], 2)
    if object_settings.get("show_labels", True):
        object_label = [None, f"{label}    {conf}"]
        text = f'{object_label[1].capitalize()}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_width = text_size[0] - 1
        #If the text is wider than the bounding box, split it into two lines: conf\nlabel
        if text_width > abs(x2 - x1):
            conf_text = f"{conf}"
            label_text = f"{label.capitalize()}"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            cv2.rectangle(overlay, (x1, y1 - 31), (x1 + label_size[0] - 1, y1 - 6), (42, 42, 42), -1)
            cv2.rectangle(overlay, (x1, y1 - 39 - conf_size[1]), (x1 + conf_size[0] - 1, y1 - 26), (42, 42, 42), -1)
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)
            cv2.putText(overlay, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)
            cv2.putText(frame, conf_text, (x1, y1 - 10 - label_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)
            cv2.putText(overlay, conf_text, (x1, y1 - 10 - label_size[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)
        else:
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)
            cv2.rectangle(overlay, (x1, y1 - 23 + text_size[1]), (x1 + text_size[0] - 1, y1 - 15 - text_size[1]), (42, 42, 42), -1)
            cv2.putText(overlay, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)
        
def create_ellipse_mask(height, width, box):
    #circumscribes an ellipse around the box
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    ellipse_width = int(box_width / 1.42)
    ellipse_height = int(box_height / 1.42)
    cv2.ellipse(mask, center, (ellipse_width, ellipse_height), 0, 0, 360, 255, -1)
    #next get the bounding box of the ellipse
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    mask_bbox = [max(center[0] - ellipse_width, 0), max(center[1] - ellipse_height, 0), min(width, center[0] + ellipse_width), min(height, center[1] + ellipse_height)]
    return mask, mask_bbox

def pixelate(image, mask, mask_bbox, block_size=9):
    x1, y1, x2, y2 = mask_bbox
    roi = image[y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    target_w = max(1, w // block_size)
    target_h = max(1, h // block_size)
    #skip pixelation if region is too small
    if w <= 1 or h <= 1:
        return image
    small = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if len(roi_mask.shape) == 2:
        roi_mask = roi_mask[:,:,np.newaxis]
    result = np.where(roi_mask > 0, pixelated, roi)
    image[y1:y2, x1:x2] = result
    return image

def apply_blur(image, boxes):
    result = image.copy()
    height, width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        #closer objects will be increasingly obscured
        mask, mask_bbox = create_ellipse_mask(height, width, [x1, y1, x2, y2])
        block_size = min(9 if abs(x2 - x1) <= 185 else 11 + (2 * ((x2 - x1 - 185) // 20)), 19)
        if abs(x2 - x1) < abs(y2 - y1) + 44 and block_size <= 11:
            block_size = 13
        result = pixelate(result, mask, mask_bbox, block_size)
    return result

def get_detections(results, class_assignments, objects_settings, original_shape, compressed_shape):
    detections = []
    bboxes = {}
    confidence_scores = {}
    objects_detected = set()
    untracked_id = -1
    for result in results:
        for detection in result.boxes:
            if int(detection.cls.item()) in class_assignments.values():
                cls_id = int(detection.cls.item())
                class_label = next(key for key, value in class_assignments.items() if value == cls_id)
                confidence_score = detection.conf.item()
                if confidence_score > objects_settings[class_label].get("confidence_threshold", 0.55):
                    x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                    bbox = rescale_bbox([x1, y1, x2, y2], original_shape, compressed_shape)
                    if objects_settings[class_label].get("enable_tracking", False):
                        yolo_id = int(detection.id.item() if detection.id is not None else untracked_id)
                        if yolo_id < 1:
                            untracked_id -= 1
                        #detections only get added to the list if they are tracked
                        detections.append((class_label, yolo_id, bbox, confidence_score))
                    else:
                        if class_label not in bboxes:
                            bboxes[class_label] = []
                        if class_label not in confidence_scores:
                            confidence_scores[class_label] = []
                        bboxes[class_label].append(bbox)
                        confidence_scores[class_label].append(confidence_score)
                    objects_detected.add(class_label)
    return detections, bboxes, objects_detected, confidence_scores

def update_tracker(objects_settings, id_tracker, detections, bboxes, confidence_scores):
    enable_tracking = any(objects_settings[obj].get("enable_tracking", False) for obj in objects_settings)
    if enable_tracking:
        for class_label in objects_settings:
            if class_label in confidence_scores and objects_settings[class_label].get("enable_tracking", False):
                del confidence_scores[class_label]
        id_tracker.update(detections)
        for class_label, yolo_id, bbox, confidence_score in detections:
            tracker_id = id_tracker.get_tracker_id(class_label, yolo_id)
            label = f"{class_label}{tracker_id}"
            bboxes[label] = bbox
            confidence_scores[label] = confidence_score
    return bboxes, confidence_scores

def get_face_box(bbox):
    #manually get the face box from the person box
    x1, y1, x2, y2 = bbox
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    high_point = min(y1, y2) + (width // 3)
    face_box = x1, high_point + (width // 5), x2, high_point - (width // 3)
    #if the height is close enough to the width, we assume the person is just walking into frame
    #or the camera is top-down and set the box as the whole person box (adding a little extra height)
    if height < (width + 44):
        face_box = x1, max(y1, y2), x2, high_point - (width // 3)
    return face_box

def blur_person_faces(bboxes, frame, blur_and_tracking, blur_tracking_info, camera_id, task_uuid):
    face_boxes = []
    for label, bbox in bboxes.items():
        if label.startswith("person"):
            if isinstance(bbox[0], list):
                for box in bbox:
                    face_boxes.append(get_face_box(box))
            else:
                face_box = get_face_box(bbox)
                face_boxes.append(face_box)
                blur_tracking_info[label] = TrackingInfo(label, face_box, 0)
    
    if "person" in blur_and_tracking:
        if camera_id not in id_trackers["blur"]:
            id_trackers["blur"][camera_id] = {}
        if task_uuid not in id_trackers["blur"][camera_id]:
            id_trackers["blur"][camera_id][task_uuid] = {}
        for label in list(id_trackers["blur"][camera_id][task_uuid].keys()):
            if label not in blur_tracking_info.keys():
                face_boxes.append(id_trackers["blur"][camera_id][task_uuid][label].bbox)
                del id_trackers["blur"][camera_id][task_uuid][label]
        id_trackers["blur"][camera_id][task_uuid] = blur_tracking_info
    return apply_blur(frame, face_boxes)

def blur_objects(needs_blur, blur_enabled_objects, tracking_enabled_objects, objects_detected, bboxes, camera_id, task_uuid, input_frame):
    if not needs_blur:
        return input_frame.copy()
    blurred_frame = input_frame.copy()
    blur_and_tracking = set(blur_enabled_objects) & set(tracking_enabled_objects)
    if any(object_cls in tracking_enabled_objects for object_cls in blur_enabled_objects):
        blur_tracking_info = {cls: {} for cls in blur_and_tracking}
    else:
        blur_tracking_info = {}
    for object_cls in (set(blur_enabled_objects) & set(objects_detected)):
        if object_cls == "person":
            blurred_frame = blur_person_faces(bboxes, blurred_frame, blur_and_tracking, blur_tracking_info, camera_id, task_uuid)
        else:
            continue
    return blurred_frame

def draw_annotations_and_overlay(annotated_frame, bboxes, colorPalette, objects_settings, confidence_scores):
    annotated_frame_with_overlay = annotated_frame.copy()
    for label, bbox in bboxes.items():
        #cls is label with the number removed from the end
        cls = ''.join([i for i in label if not i.isdigit()])
        if isinstance(bbox[0], list):
            for i in range(len(bbox)):
                draw_annotation(annotated_frame, annotated_frame_with_overlay, label, f"{confidence_scores[cls][i]:.2f}", bbox[i], colorPalette, objects_settings[cls], cls)
        else:
            draw_annotation(annotated_frame, annotated_frame_with_overlay, label, f"{confidence_scores[label]:.2f}", bbox, colorPalette, objects_settings[cls], cls)
    alpha = 0.35
    result_image = cv2.addWeighted(annotated_frame_with_overlay, alpha, annotated_frame, 1 - alpha, 0)
    return result_image

def get_total_objects(bboxes):
    total_objects = 0
    for _, bbox in bboxes.items():
        total_objects += len(bbox) if isinstance(bbox[0], list) else 1
    return total_objects

def resize_for_inference(input_frame, target_shape):
    #Resize the input frame to the target shape for faster inference
    return cv2.resize(input_frame, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)

def rescale_bbox(bbox, original_shape, compressed_shape):
    x1, y1, x2, y2 = bbox
    #Calculate width and height scaling factors directly
    width_scale = original_shape[1] / compressed_shape[1]
    height_scale = original_shape[0] / compressed_shape[0]
    #Apply scaling to each coordinate
    scaled_x1 = int(x1 * width_scale)
    scaled_y1 = int(y1 * height_scale)
    scaled_x2 = int(x2 * width_scale)
    scaled_y2 = int(y2 * height_scale)
    return [scaled_x1, scaled_y1, scaled_x2, scaled_y2]

def detect_objects(camera_id, task_settings, input_frame, frame_shape, roi=None):
    if roi is not None:
        input_frame = input_frame[roi[1]:roi[3], roi[0]:roi[2]]
    objects_settings = task_settings.get("objects_to_detect")
    objects = list(objects_settings.keys())
    task_uuid = task_settings.get("task_uuid", "default")
    #annotated_objects holds all members of objects where their object_settings[show_boxes] or object_settings[show_labels] is True

    annotated_objects = [obj for obj in objects if objects_settings[obj].get("show_boxes", False) or objects_settings[obj].get("show_labels", False)]
    blur_enabled_objects = [obj for obj in objects if objects_settings[obj].get("enable_blur", False)]
    tracking_enabled_objects = [obj for obj in objects if objects_settings[obj].get("enable_tracking", False)]
    
    class_assignments = get_class_assignments(objects)
    
    enable_tracking = any(objects_settings[obj].get("enable_tracking", False) for obj in objects_settings)
    #needs_blur is true if both objects has any supported_blur_classes and if for any of the supported_blur_classes in objects_settings, enable_blur is True

    needs_blur = any(obj in blur_enabled_objects for obj in supported_blur_classes)
    
    model, id_tracker = get_model_and_tracker(camera_id, task_uuid, class_assignments, enable_tracking)
    
    #Resize input_frame to smaller dimensions for faster inference
    compressed_shape = [360, 640]
    compressed_input_frame = resize_for_inference(input_frame, compressed_shape)
    
    if enable_tracking:
        results = model.track(compressed_input_frame, persist=True, verbose=False)
    else:
        results = model(compressed_input_frame, verbose=False)

    colorPalette = get_colors(camera_id, task_uuid, annotated_objects, input_frame, frame_shape)
    
    detections, bboxes, objects_detected, confidence_scores = get_detections(results, class_assignments, objects_settings, frame_shape, compressed_shape)
    
    bboxes, confidence_scores = update_tracker(objects_settings, id_tracker, detections, bboxes, confidence_scores)
    
    blurred_frame = blur_objects(needs_blur, blur_enabled_objects, tracking_enabled_objects, objects_detected, bboxes, camera_id, task_uuid, input_frame)
    result_image = draw_annotations_and_overlay(blurred_frame, bboxes, colorPalette, objects_settings, confidence_scores)

    return result_image, bboxes, list(objects_detected), get_total_objects(bboxes)

if __name__ == '__main__':
    import os
    import time
    import sys
    import cv2
    from datetime import datetime
    from collections import deque
    
    #Add parent directory to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(parent_dir)
    
    #Save current working directory
    original_dir = os.getcwd()
    
    try:
        #Change directory to object_detection folder so it can find its assets
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        #Create a test environment
        os.environ['CB_SYSTEM_KEY'] = "test_system_key"
        
        #Test video path - update this path to your test video
        video_path = os.path.abspath('/path/to/your/video.mp4')
        
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
        
        #Create object detection settings
        object_detection_settings = {
            "objects_to_detect": {
                "person": {
                    "enable_tracking": True,
                    "enable_blur": False,
                    "show_boxes": True,
                    "show_labels": True,
                    "confidence_threshold": 0.65
                }
            }
        }
        
        #Create a window to display the frames
        cv2.namedWindow('Object Detection Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection Test', width // 2, height // 2)
        
        frame_count = 0
        processed_count = 0
        
        #For calculating average processing time
        process_times = deque(maxlen=10)
        calibration_complete = False
        
        print("Testing object detection...")
        print("Calibrating frame skip rate with first 10 frames...")
        
        #Variable for printing detection info (limit to once every 5 seconds)
        last_print_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            #Process only selected frames based on frame_skip
            #During calibration, process all frames
            if not calibration_complete or frame_count % frame_skip == 0:
                process_start = time.time()
                
                try:
                    #Detect objects in the frame
                    annotated_frame, bboxes, objects_detected, total_objects = detect_objects(
                        "test_camera", object_detection_settings, frame.copy(), frame_shape
                    )
                    
                    #Print detection info (but not more than once every 5 seconds)
                    current_time = time.time()
                    if current_time - last_print_time >= 5.0:
                        print(f"\nFrame {frame_count} - Detected objects: {objects_detected}")
                        print(f"Total objects detected: {total_objects}")
                        
                        #Print bounding box data
                        print("Bounding boxes:")
                        for label, bbox in bboxes.items():
                            print(f"  {label}: {bbox}")
                        
                        last_print_time = current_time
                    
                    #Display the frame
                    cv2.imshow('Object Detection Test', annotated_frame)
                    
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
                    import traceback
                    traceback.print_exc()
                    process_times.append(0.1)  #Use a default value for errors
            
            #Check for key press to quit
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
        print(f"\nTest completed. Processed {processed_count} out of {frame_count} frames.")
        if len(process_times) > 0:
            print(f"Final average processing time: {sum(process_times)/len(process_times):.6f} seconds")
    
    except Exception as e:
        print(f"Test error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        #Change back to original directory
        os.chdir(original_dir)
