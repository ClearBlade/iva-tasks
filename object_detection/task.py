import base64

import cv2
import numpy as np
import onnxruntime as rt
from yolo_utils import (get_class_assignments, get_preprocessed_frame,
                        post_process_pipeline)


def convertB64ToFrame(b64_string):
    jpg_original = base64.b64decode(b64_string)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
    return img

def convertFrameToB64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def load_model(model_path):
    """
    Load the model from the given path.

    Args:
        model_path (str): The path to the model.

    Returns:
        None
    """
    if not model_path:
        raise Exception("No model file path provided!")

    providers = ["CUDAExecutionProvider"]
    sess = rt.InferenceSession(model_path, providers=providers)

    print(f"Onnxruntime Device: {rt.get_device()}")

    outputs = sess.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = sess.get_inputs()[0].name
    return input_name, output_names, sess

def run_model(sess, output_names, input_name, image_data):
    """
    runs yolov4 model on an image

    Arguments: session, names, image
    Returns: model detections
    """

    detections = sess.run(output_names, {input_name: image_data})
    return detections

def detect_objects(input_frame, task_settings):
    """
    Task function to detect objects.

    Args:
        input_frame (dict): Image Array.

    Returns:
        image_with_bboxes (dict): Image Array with detected objects.
        bboxes (list): List of bounding boxes.
    """
    objects = task_settings.get("objects_to_detect")
    roi_coords = task_settings.get("roi_coords")
    confidence_threshold = task_settings.get("confidence_threshold", 50)

    class_assignment = get_class_assignments(objects)
    input_size, original_image_size, original_image, image_data = get_preprocessed_frame(input_frame, roi_coords)
    
    input_name, output_names, sess = load_model('assets/tiny-yolov4.onnx')
    detections = run_model(sess, output_names, input_name, image_data)
    
    bboxes, image_with_bboxes = post_process_pipeline(detections, original_image_size, input_size, original_image, class_assignment, confidence_threshold/100)
    
    return image_with_bboxes, np.array(bboxes).tolist()

if __name__ == '__main__':
    input_frame = cv2.imread('assets/test.png')
    task_settings = {
        "objects_to_detect": ["person"],
        "confidence_threshold": 50
    }
    image_with_bboxes, _ = detect_objects(input_frame, task_settings)
    cv2.imwrite('output.jpg', image_with_bboxes)