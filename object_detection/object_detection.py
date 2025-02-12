import base64
import cv2
import numpy as np
import time
from ultralytics import YOLO

#Global dictionaries to store models and trackers for each camera
models = {}
id_trackers = {}
colors = {}

class IDTracker:
    def __init__(self, classifications, max_disappeared=2):
        self.classifications = classifications
        self.max_disappeared = max_disappeared
        self.tracker_ids = {cls: {} for cls in classifications}
        self.disappeared = {cls: {} for cls in classifications}

    def update(self, detections):
        current_yolo_ids = {cls: set() for cls in self.classifications}
        # print("Yolo IDs:", detections)

        for cls, yolo_id in detections:
            current_yolo_ids[cls].add(yolo_id)

            if yolo_id in self.tracker_ids[cls]:
                tracker_id = self.tracker_ids[cls][yolo_id]
                self.disappeared[cls][tracker_id] = 0
            else:
                if yolo_id != -1:
                    new_tracker_id = self.get_next_available_id(cls)
                    self.tracker_ids[cls][yolo_id] = new_tracker_id
                    self.disappeared[cls][new_tracker_id] = 0
                else:
                    pass

        for cls in self.classifications:
            for yolo_id, tracker_id in list(self.tracker_ids[cls].items()):
                if yolo_id not in current_yolo_ids[cls]:
                    self.disappeared[cls][tracker_id] += 1
                    if self.disappeared[cls][tracker_id] > self.max_disappeared:
                        del self.tracker_ids[cls][yolo_id]
                        del self.disappeared[cls][tracker_id]

        return self.tracker_ids

    def get_next_available_id(self, cls):
        used_ids = set(self.tracker_ids[cls].values())
        tracker_id = 1
        while tracker_id in used_ids:
            tracker_id += 1
        return tracker_id

    def get_tracker_id(self, cls, yolo_id):
        return self.tracker_ids[cls].get(yolo_id, None)

def convertB64ToFrame(b64_string):
    jpg_original = base64.b64decode(b64_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    return img

def convertFrameToB64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def get_model_and_tracker(camera_id, class_assignments, enable_tracking):
    if enable_tracking:
        if camera_id not in models:
            models[camera_id] = YOLO("assets/yolo11s.onnx", task='detect')
            id_trackers[camera_id] = IDTracker(class_assignments.keys())
        return models[camera_id], id_trackers[camera_id]
    else:
        if camera_id not in models:
            models[camera_id] = YOLO("assets/yolo11s.onnx", task='detect')
        return models[camera_id], None

def get_colors(camera_id, class_assignments, frame, frame_shape):
    if camera_id not in colors:
        box_coords = (0, 0, frame_shape[1], frame_shape[0])
        colors[camera_id] = choose_colors(frame, box_coords, class_assignments)
    return colors[camera_id]

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

def detect_objects(camera_id, task_settings, input_frame, frame_shape):
    objects = task_settings.get("objects_to_detect")
    class_assignments = get_class_assignments(objects)
    enable_tracking = task_settings.get("object_tracking", False)

    model, id_tracker = get_model_and_tracker(camera_id, class_assignments, enable_tracking) #id_tracker is None if tracking is disabled
    
    confidence_threshold = task_settings.get("confidence_threshold", 0.65)
    
    if enable_tracking:
        results = model.track(input_frame, persist=True, verbose=False)
    else:
        results = model(input_frame, verbose=False)
    
    annotated_frame = input_frame.copy()
    annotated_frame_with_overlay = annotated_frame.copy()
    detections = []
    bboxes = {}
    objects_detected = set()
    colorPalette = get_colors(camera_id, objects, input_frame, frame_shape)
    
    for result in results:
        for detection in result.boxes:
            if detection.conf.item() > confidence_threshold and int(detection.cls.item()) in class_assignments.values():
                cls_id = int(detection.cls.item())
                class_label = next(key for key, value in class_assignments.items() if value == cls_id)
                x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                if enable_tracking:
                    yolo_id = int(detection.id.item() if detection.id is not None else -1)
                    detections.append((class_label, yolo_id))
                    tracker_id = id_tracker.get_tracker_id(class_label, yolo_id)
                    label = f"{class_label}{tracker_id if tracker_id else ''}"
                    bboxes[label] = [x1, y1, x2, y2]
                else:
                    if class_label not in bboxes:
                        bboxes[class_label] = []
                    bboxes[class_label].append([x1, y1, x2, y2])
                
                objects_detected.add(class_label)
                #Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), colorPalette[class_label], 2)
                cv2.rectangle(annotated_frame_with_overlay, (x1, y1), (x2, y2), colorPalette[class_label], 2)
                text = f'{class_label.capitalize()}{tracker_id if enable_tracking and tracker_id else ""} {detection.conf.item():.2f}'
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[class_label], 2, lineType = cv2.LINE_AA)
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                
                cv2.rectangle(annotated_frame_with_overlay, (x1, y1 - 23 + text_size[1]), (x1 + text_size[0], y1 - 15 - text_size[1]), (42, 42, 42), -1)
                cv2.putText(annotated_frame_with_overlay, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[class_label], 2, lineType = cv2.LINE_AA)
    
    alpha = 0.35
    result_image = cv2.addWeighted(annotated_frame_with_overlay, alpha, annotated_frame, 1 - alpha, 0)

    if enable_tracking:
        id_tracker.update(detections)
        total_objects = len(bboxes)
    else:
        total_objects = sum(len(boxes) if isinstance(boxes, list) else 1 for boxes in bboxes.values())
    


    #Uncomment the lines below to save the annotated frame to a file
    #frame_name = time.strftime("%Y%m%d-%H%M%S") + str(time.time() % 1)[1:3]
    #cv2.imwrite(f"{frame_name}.jpg", result_image)
    #print("image saved as: ", f"{frame_name}.jpg")
    
    return result_image, bboxes, list(objects_detected), total_objects




if __name__ == '__main__':
    #Test code here if needed
    pass