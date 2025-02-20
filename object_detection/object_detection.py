import cv2
import numpy as np
import time
from ultralytics import YOLO

models = {}
id_trackers = {}
colors = {}
supported_blur_classes = ["person"]

class TrackingInfo:
    def __init__(self, tracker_id, bbox, yolo_id=None):
        self.tracker_id = tracker_id
        self.bbox = bbox
        self.yolo_id = yolo_id
        
class IDTracker:
    def __init__(self, classifications, distance_threshold=200):
        self.classifications = classifications
        self.distance_threshold = distance_threshold
        self.tracking_info = {cls: {} for cls in classifications}
        self.untracked_objects = {cls: {} for cls in classifications}

    def update(self, detections):
        current_yolo_ids = {cls: set() for cls in self.classifications}
        untracked_detections = []
        new_tracking_info = {cls: {} for cls in self.classifications}
        #First, handle all tracked objects
        for cls, yolo_id, bbox, _ in detections:
            if yolo_id >= 0:
                current_yolo_ids[cls].add(yolo_id)
                if yolo_id in self.tracking_info[cls]:
                    tracker_id = self.tracking_info[cls][yolo_id].tracker_id
                else:
                    tracker_id = self.handle_new_tracked_object(cls, yolo_id, bbox)
                new_tracking_info[cls][yolo_id] = TrackingInfo(tracker_id, bbox, yolo_id)
            else:
                untracked_detections.append((cls, yolo_id, bbox))
        #Handle untracked objects
        self.update_pretracking(untracked_detections, new_tracking_info)
        #Update tracking_info
        self.tracking_info = new_tracking_info
        return self.tracking_info

    def update_pretracking(self, untracked_detections, new_tracking_info):
        for cls, yolo_id, bbox in untracked_detections:
            matched = False
            for tracker_id, info in list(self.untracked_objects[cls].items()):
                distance = get_distance_between_objects(bbox, info.bbox)
                if distance <= self.distance_threshold:
                    new_tracking_info[cls][yolo_id] = TrackingInfo(tracker_id, bbox, yolo_id)
                    matched = True
                    break
            if not matched:
                tracker_id = self.get_next_available_id(cls)
                new_tracking_info[cls][yolo_id] = TrackingInfo(tracker_id, bbox, yolo_id)
        #Update untracked_objects with new untracked detections
        self.untracked_objects = {cls: {} for cls in self.classifications}
        for cls, yolo_id in new_tracking_info.items():
            for info in new_tracking_info[cls].values():
                self.untracked_objects[cls][info.tracker_id] = info

    def handle_new_tracked_object(self, cls, yolo_id, bbox):
        for tracker_id, info in self.untracked_objects[cls].items():
            distance = get_distance_between_objects(bbox, info.bbox)
            if distance <= self.distance_threshold:
                return tracker_id
        return self.get_next_available_id(cls)

    def get_next_available_id(self, cls):
        used_ids = set(info.tracker_id for info in self.tracking_info[cls].values())
        used_ids.update(info.tracker_id for info in self.untracked_objects[cls].values())
        tracker_id = 1
        while tracker_id in used_ids:
            tracker_id += 1
        return tracker_id

    def free_up_tracker_ids(self, new_tracking_info):
        for cls in self.classifications:
            current_tracker_ids = set(info.tracker_id for info in new_tracking_info[cls].values())
            self.untracked_objects[cls] = {
                tracker_id: info
                for tracker_id, info in self.untracked_objects[cls].items()
                if tracker_id in current_tracker_ids
            }

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

def get_face_model(blur_faces):
    if blur_faces:
        if "facial_anonymization" not in models:
            models["facial_anonymization"] = YOLO("assets/yolo11n-head.onnx", task='detect')
        return models["facial_anonymization"]
    return None

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

def draw_annotation(frame, overlay, label, bbox, colorPalette, object_settings, cls):
    x1, y1, x2, y2 = bbox
    if object_settings.get("show_boxes", True):
        cv2.rectangle(frame, (x1, y1), (x2, y2), colorPalette[cls], 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), colorPalette[cls], 2)
    if object_settings.get("show_labels", True):
        text = f'{label.capitalize()}   '
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.rectangle(overlay, (x1, y1 - 23 + text_size[1]), (x1 + text_size[0] - 45, y1 - 15 - text_size[1]), (42, 42, 42), -1)
        cv2.putText(overlay, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colorPalette[cls], 2, lineType=cv2.LINE_AA)


def create_ellipse_mask(height, width, box):
    #draws an ellipse over the bounding box
    mask = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = map(int, box)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    ellipse_width = int((x2 - x1) / 1.875)
    ellipse_height = int((y2 - y1) / 1.875)
    cv2.ellipse(mask, center, (ellipse_width, ellipse_height), 0, 0, 360, 255, -1)
    return mask

def pixelate(image, mask, avg_color, block_size=9):
    h, w = image.shape[:2]
    base = np.full((h, w, 3), avg_color, dtype=np.uint8)
    masked_image = np.where(mask[:,:,np.newaxis] > 0, image, base)
    small = cv2.resize(masked_image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def create_gradient(height, width, mask, avg_color):
    base = np.full((height, width, 3), avg_color, dtype=np.uint8)
    y, x = np.ogrid[:height, :width]
    center = (width // 2, height // 2)
    distances = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    max_distance = np.max(distances[mask > 0])
    normalized_distances = distances / max_distance
    gradient = (1 - normalized_distances[:,:,np.newaxis]) * avg_color
    gradient = np.where(mask[:,:,np.newaxis] > 0, gradient, base)
    return gradient.astype(np.uint8)

def apply_blur(image, boxes):
    result = image.copy()
    height, width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        box_xywh = [x1, y1, x2-x1, y2-y1]
        avg_color = get_average_color(image, box_xywh)
        mask = create_ellipse_mask(height, width, [x1, y1, x2, y2])
        pixelated = pixelate(image, mask, avg_color)
        gradient = create_gradient(height, width, mask, avg_color)
        combined = cv2.addWeighted(pixelated, 0.5, gradient, 0.5, 0)
        result = np.where(mask[:,:,np.newaxis] > 0, combined, result)
    return result

def detect_faces(face_model, image, confidence_threshold=0.001):
    results = face_model(image, verbose=False)
    face_boxes = []
    for result in results:
        for detection in result.boxes:
            if detection.conf.item() > confidence_threshold:
                x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                face_boxes.append([x1, y1, x2, y2])
    return face_boxes

def get_detections(results, class_assignments, objects_settings):
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
                if confidence_score > objects_settings[class_label].get("confidence_threshold", 0.65):
                    x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                    bbox = [x1, y1, x2, y2]
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
        confidence_scores = {}
        id_tracker.update(detections)
        for class_label, yolo_id, bbox, confidence_score in detections:
            tracker_id = id_tracker.get_tracker_id(class_label, yolo_id)
            label = f"{class_label}{tracker_id}"
            bboxes[label] = bbox
            confidence_scores[label] = confidence_score
    return bboxes, confidence_scores

def blur_objects(needs_blur, objects, objects_detected, bboxes, input_frame):
    if not needs_blur:
        return input_frame.copy()
    if "person" in objects:
        face_model = get_face_model(needs_blur)
        detects_people = "person" in objects
        if detects_people and "person" in objects_detected:
            face_boxes = []
            for label, bbox in bboxes.items():
                if label.startswith("person"):
                    if isinstance(bbox[0], list):
                        for box in bbox:
                            x1, y1, x2, y2 = box
                            person_roi = input_frame[y1:y2, x1:x2]
                            face_boxes_in_roi = detect_faces(face_model, person_roi)
                            for face_box in face_boxes_in_roi:
                                face_boxes.append([x1 + face_box[0], y1 + face_box[1], x1 + face_box[2], y1 + face_box[3]])
                    else:
                        x1, y1, x2, y2 = bbox
                        person_roi = input_frame[y1:y2, x1:x2]
                        face_boxes_in_roi = detect_faces(face_model, person_roi)
                        for face_box in face_boxes_in_roi:
                            face_boxes.append([x1 + face_box[0], y1 + face_box[1], x1 + face_box[2], y1 + face_box[3]])
            blurred_frame = apply_blur(input_frame, face_boxes)
        elif not detects_people:
            face_boxes = detect_faces(face_model, input_frame)
            blurred_frame = apply_blur(input_frame, face_boxes)
        else:
            blurred_frame = input_frame.copy()
    #elif "other_class" in objects:
        #only person is supported for now
    else:
        blurred_frame = input_frame.copy()
    return blurred_frame

def draw_annotations_and_overlay(annotated_frame, bboxes, colorPalette, objects_settings, confidence_scores):
    annotated_frame_with_overlay = annotated_frame.copy()
    for label, bbox in bboxes.items():
        #cls is label with the number removed from the end
        cls = ''.join([i for i in label if not i.isdigit()])
        #object label is label if object_settings[cls][show_labels] is True, otherwise it's ''
        object_label = label if objects_settings[cls].get("show_labels", True) else ''
        if isinstance(bbox[0], list):
            for i in range(len(bbox)):
                #add 4 spaces then confidence score to object label if it is not ''
                object_label = f"{label}    {confidence_scores[cls][i]:.2f}" if object_label else ''
                draw_annotation(annotated_frame, annotated_frame_with_overlay, object_label, bbox[i], colorPalette, objects_settings[cls], cls)
        else:
            object_label = f"{label}    {confidence_scores[label]:.2f}" if object_label else ''
            draw_annotation(annotated_frame, annotated_frame_with_overlay, object_label, bbox, colorPalette, objects_settings[cls], cls)
    alpha = 0.35
    result_image = cv2.addWeighted(annotated_frame_with_overlay, alpha, annotated_frame, 1 - alpha, 0)
    return result_image

def get_total_objects(bboxes):
    total_objects = 0
    for label, bbox in bboxes.items():
        total_objects += len(bbox) if isinstance(bbox[0], list) else 1
    return total_objects

def detect_objects(camera_id, task_settings, input_frame, frame_shape, roi=None):
    if roi is not None:
        input_frame = input_frame[roi[1]:roi[3], roi[0]:roi[2]]
    objects_settings = task_settings.get("objects_to_detect")
    objects = list(objects_settings.keys())
    #annotated_objects holds all members of objects where their object_settings[show_boxes] or object_settings[show_labels] is True
    annotated_objects = [obj for obj in objects if objects_settings[obj].get("show_boxes", False) or objects_settings[obj].get("show_labels", False)]
    blur_enabled_objects = [obj for obj in objects if objects_settings[obj].get("enable_blur", False)]
    class_assignments = get_class_assignments(objects)
    enable_tracking = any(objects_settings[obj].get("enable_tracking", False) for obj in objects_settings)
    #needs_blur is true if both objects has any supported_blur_classes and if for any of the supported_blur_classes in objects_settings, enable_blur is True
    needs_blur = any(obj in blur_enabled_objects for obj in supported_blur_classes)
    model, id_tracker = get_model_and_tracker(camera_id, class_assignments, enable_tracking)    
    if enable_tracking:
        results = model.track(input_frame, persist=True, verbose=False)
    else:
        results = model(input_frame, verbose=False)
    colorPalette = get_colors(camera_id, annotated_objects, input_frame, frame_shape)
    detections, bboxes, objects_detected, confidence_scores = get_detections(results, class_assignments, objects_settings)
    bboxes, confidence_scores = update_tracker(objects_settings, id_tracker, detections, bboxes, confidence_scores)
    blurred_frame = blur_objects(needs_blur, blur_enabled_objects, objects_detected, bboxes, input_frame) #if blur_faces is False, this function will return the input_frame as is
    result_image = draw_annotations_and_overlay(blurred_frame, bboxes, colorPalette, objects_settings, confidence_scores)
    total_objects = get_total_objects(bboxes)
    return result_image, bboxes, list(objects_detected), total_objects

if __name__ == '__main__':
    #Test code here if needed
    pass