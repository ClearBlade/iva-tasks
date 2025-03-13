import cv2
from shapely.geometry import LineString, Polygon, Point
from collections import defaultdict
import numpy as np

DIRECTION_A_TO_B = "A_TO_B"
DIRECTION_B_TO_A = "B_TO_A"

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
    #Test code here if needed
    pass
