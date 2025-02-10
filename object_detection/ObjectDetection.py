import base64
import cv2
import numpy as np
import time
from ultralytics import YOLO

#Global dictionaries to store models and trackers for each camera
models = {}
id_trackers = {}

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

def get_model_and_tracker(camera_id, class_assignments):
    if camera_id not in models:
        models[camera_id] = YOLO("assets/yolo11s.onnx", task='detect')
        id_trackers[camera_id] = IDTracker(class_assignments.keys())
    return models[camera_id], id_trackers[camera_id]

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

def detect_objects(input_frame, camera_id, task_settings):
    objects = task_settings.get("objects_to_detect")
    class_assignments = get_class_assignments(objects)
    
    model, id_tracker = get_model_and_tracker(camera_id, class_assignments)
    
    enable_tracking = task_settings.get("object_tracking", True)
    confidence_threshold = task_settings.get("confidence_threshold", 0.65)
    
    results = model.track(input_frame, persist=True, verbose=False)
    
    annotated_frame = input_frame.copy()
    detections = []
    bboxes = {}
    objects_detected = set()
    
    for result in results:
        for detection in result.boxes:
            if detection.conf.item() > confidence_threshold and int(detection.cls.item()) in class_assignments.values():
                cls_id = int(detection.cls.item())
                yolo_id = int(detection.id.item() if detection.id is not None else -1)
                class_label = next(key for key, value in class_assignments.items() if value == cls_id)
                detections.append((class_label, yolo_id))
                
                x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
                
                if enable_tracking:
                    tracker_id = id_tracker.get_tracker_id(class_label, yolo_id)
                    label = f"{class_label}{tracker_id if tracker_id else ''}"
                    bboxes[label] = [x1, y1, x2, y2]
                else:
                    if class_label not in bboxes:
                        bboxes[class_label] = []
                    bboxes[class_label].append([x1, y1, x2, y2])
                
                objects_detected.add(class_label)
                
                #Draw bounding box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = f'{class_label.capitalize()}{tracker_id if enable_tracking and tracker_id else ""} {detection.conf.item():.2f}'
                cv2.putText(annotated_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if enable_tracking:
        id_tracker.update(detections)
        total_objects = len(bboxes)
    else:
        total_objects = sum(len(boxes) if isinstance(boxes, list) else 1 for boxes in bboxes.values())
    


    #Uncomment the lines below to save the annotated frame to a file
    #frame_name = time.strftime("%Y%m%d-%H%M%S") + str(time.time() % 1)[1:3]
    #cv2.imwrite(f"{frame_name}.jpg", annotated_frame)
    #print("image saved as: ", f"{frame_name}.jpg")    
    
    return annotated_frame, bboxes, list(objects_detected), total_objects




if __name__ == '__main__':
    #Test code here if needed
    pass