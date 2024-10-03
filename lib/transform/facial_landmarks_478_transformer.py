import cv2
import mediapipe as mp
import numpy as np

circleDrawingSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

LEFT_IRIS = [ 474,475, 476, 477 ]
RIGHT_IRIS = [ 469, 470, 471, 472 ]

LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398 ]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246 ]

INTERNAL_LIPS = [ 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78 ]
EXTERNAL_LIPS = [ 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61 ]

COLOR = (255, 0, 255)
COLOR_IRIS = (255, 0, 0)

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

SCALE_W = 0.35 #178/512
SCALE_H =0.43 #218/512
class Ellipse:
    def __init__(self, points):
        self.points = points
        self.axis = self.calculate_axis()
        self.centroid = self.calculate_centroid()

    def __str__(self):
        attributes = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"Ellipse({attributes})"
    
    def calculate_axis(self):
        axis1 = self.points[3][1] - self.points[1][1]
        axis2 = self.points[0][0] - self.points[2][0]
        return (axis2*SCALE_H, axis1*SCALE_H)
    
    def calculate_centroid(self):
        mean_x = sum(point[0] for point in self.points) / len(self.points)
        mean_y = sum(point[1] for point in self.points) / len(self.points)
        return (mean_x, mean_y)

class Eyes:
    def __init__(self, points1, points2):
        self.ellipse_left = Ellipse(points1)
        self.ellipse_right = Ellipse(points2)

    def __str__(self):
        attributes = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"Ellipse({attributes})"

    def draw_on_image(self, image, color=(0, 255, 0), thickness=2):
        center_left = tuple(map(int, self.ellipse_left.centroid))
        axes_length_left = tuple(map(int, self.ellipse_left.axis))
        center_right = tuple(map(int, self.ellipse_right.centroid))
        axes_length_right = tuple(map(int, self.ellipse_right.axis))
        
        cv2.ellipse(image, center_left, axes_length_left, 0, 0, 360, color, thickness)
        cv2.ellipse(image, center_right, axes_length_right, 0, 0, 360, color, thickness)

def config(mesh_congiguration, point_image, face_landmarks, pic, results):
    mesh_number = mesh_congiguration.split('_')[0]
    #print("MESH NUMBER: ", mesh_number)
    if mesh_number == '00':
        draw_face_keypoints(point_image, face_landmarks)
        #draw_iris_mesh(point_image, face_landmarks)

    elif  mesh_number == '02':
        draw_face_tesselation(point_image, face_landmarks)
        point_image = draw_eye_region(point_image, pic.shape, results, mesh_number)

    elif mesh_number == '03':
        draw_face_keypoints(point_image, face_landmarks)
        point_image = draw_eye_region(point_image, pic.shape, results, mesh_number)
    
    elif mesh_number == '04':
        draw_face_tesselation(point_image, face_landmarks)
        point_image = draw_eye_region(point_image, pic.shape, results, mesh_number)

    elif mesh_number == '05':#TODO fix this
        draw_face_tesselation(point_image, face_landmarks)
        point_image = draw_eye_region_colorful(point_image, pic.shape, results, mesh_number)

    else:
        draw_face_keypoints(point_image, face_landmarks)
    return point_image

    #draw_face_tesselation(point_image, face_landmarks)        
    #draw_face_contour(point_image, face_landmarks)
    #point_image = draw_eye_region(point_image, pic.shape, results)
    #draw_iris(point_image, pic.shape, results)

def draw_countour_iris(img, shape, results):
    height, width, channels = shape
    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    eyes = Eyes(mesh_points[LEFT_IRIS], mesh_points[RIGHT_IRIS])

    center_left = tuple(map(int, eyes.ellipse_left.centroid))
    axes_length_left = tuple(map(int, eyes.ellipse_left.axis))
    center_right = tuple(map(int, eyes.ellipse_right.centroid))
    axes_length_right = tuple(map(int, eyes.ellipse_right.axis))

    countour = cv2.polylines(img, [mesh_points[LEFT_EYE ]], True, RED, 2, cv2.LINE_AA)
    countour = cv2.polylines(countour, [mesh_points[RIGHT_EYE]], True, RED, 2, cv2.LINE_AA)
    countour = cv2.ellipse(countour,  center_left, axes_length_left, 0, 0, 360, RED, 2)#TODO ADD FLAG
    countour = cv2.ellipse(countour, center_right, axes_length_right, 0, 0, 360, RED, 2)
    return countour

def draw_eye_region(img, shape, results, mesh_number, fill=True):
    height, width, channels = shape
    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    img_copy = img.copy()
    if fill:
        mask =cv2.fillPoly(img, [mesh_points[LEFT_EYE]], COLOR)
        mask = cv2.fillPoly(img, [mesh_points[RIGHT_EYE]], COLOR)
        if mesh_number == '04':
            return draw_iris(mask, img_copy, shape, results)
        else:
            return draw_iris_circle(mask, img_copy, shape, results)

    else:
        cv2.polylines(img, [mesh_points[LEFT_EYE ]], True, COLOR, 1, cv2.LINE_AA)
        cv2.polylines(img, [mesh_points[RIGHT_EYE]], True, COLOR, 1, cv2.LINE_AA)
    cv2.polylines(img, [mesh_points[INTERNAL_LIPS]], True, COLOR, 1, cv2.LINE_AA)

def draw_eye_region_colorful(img, shape, results, mesh_number, fill=True):
    height, width, channels = shape
    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    img_copy = img.copy()
    if fill:
        mask = cv2.polylines(img, [mesh_points[INTERNAL_LIPS]], True, RED, 2, cv2.LINE_AA)
        img_copy = mask.copy()
        mask = cv2.fillPoly(img, [mesh_points[LEFT_EYE]], GREEN)
        mask = cv2.fillPoly(img, [mesh_points[RIGHT_EYE]], GREEN)
        return draw_iris_colorful(mask, img_copy, shape, results)
    else:#TODO check if still necessary
        cv2.polylines(img, [mesh_points[LEFT_EYE ]], True, (0,0,255), 3, cv2.LINE_AA)
        cv2.polylines(img, [mesh_points[RIGHT_EYE]], True, (0,0,255), 3, cv2.LINE_AA)
    cv2.polylines(img, [mesh_points[INTERNAL_LIPS]], True, (0,0,255), 3, cv2.LINE_AA)

def draw_face_contour(img, face_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())

def draw_iris_circle(mask, img, shape, results):
    height, width, channels = shape
    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])

    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
    # turn center points into np array 
    center_left = np.array([l_cx, l_cy], dtype=np.int32)
    center_right = np.array([r_cx, r_cy], dtype=np.int32)

    iris=cv2.circle(img, center_left, int(l_radius), COLOR, -1, cv2.LINE_AA)
    iris=cv2.circle(iris, center_right, int(r_radius), COLOR, -1, cv2.LINE_AA)    
    bitwiseAnd = cv2.bitwise_and(mask, iris)
    return bitwiseAnd

def draw_iris(mask, img, shape, results):
    height, width, channels = shape

    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    eyes = Eyes(mesh_points[LEFT_IRIS], mesh_points[RIGHT_IRIS])

    center_left = tuple(map(int, eyes.ellipse_left.centroid))
    #center_left[0] = center_left[0] - 30 # TODO desenhando iris errada
    axes_length_left = tuple(map(int, eyes.ellipse_left.axis))
    center_right = tuple(map(int, eyes.ellipse_right.centroid))
    axes_length_right = tuple(map(int, eyes.ellipse_right.axis))

    if axes_length_left[0] > 0 and axes_length_left[1] > 0 and axes_length_right[0] > 0 and axes_length_right[1] > 0:
        iris=cv2.ellipse(img,  center_left, axes_length_left, 0, 0, 360, COLOR, -1)
        iris=cv2.ellipse(iris, center_right, axes_length_right, 0, 0, 360, COLOR, -1)
        bitwiseAnd = cv2.bitwise_and(mask, iris)
        return bitwiseAnd
    else:
        return mask


def draw_iris_colorful(mask, img, shape, results):
    height, width, channels = shape

    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    eyes = Eyes(mesh_points[LEFT_IRIS], mesh_points[RIGHT_IRIS])

    center_left = tuple(map(int, eyes.ellipse_left.centroid))
    #center_left[0] = center_left[0] - 30 # TODO desenhando iris errada
    axes_length_left = tuple(map(int, eyes.ellipse_left.axis))
    center_right = tuple(map(int, eyes.ellipse_right.centroid))
    axes_length_right = tuple(map(int, eyes.ellipse_right.axis))

    if axes_length_left[0] > 0 and axes_length_left[1] > 0 and axes_length_right[0] > 0 and axes_length_right[1] > 0:
        iris=cv2.ellipse(img,  center_left, axes_length_left, 0, 0, 360, GREEN, -1)
        iris=cv2.ellipse(iris, center_right, axes_length_right, 0, 0, 360, GREEN, -1)
    else:
        return mask
    bitwiseAnd = cv2.bitwise_and(mask, iris)
    if True:
        lower_green = np.array([0,200,0])
        upper_green = np.array([0,255,0])
        green_mask = cv2.inRange(bitwiseAnd, lower_green, upper_green)
        img_green_mask = cv2.bitwise_and(bitwiseAnd, bitwiseAnd, mask=green_mask)
        mask_2 = cv2.fillPoly(img, [mesh_points[LEFT_EYE]], BLUE)
        mask_2 = cv2.fillPoly(img, [mesh_points[RIGHT_EYE]], BLUE)
        #bitwiseAnd = cv2.bitwise_and(mask_2, iris)
        bitwiseOR = cv2.bitwise_or(mask_2, img_green_mask)
        countour = draw_countour_iris(img, shape, results)
        countour_OR = cv2.bitwise_or(bitwiseOR, countour)

        lower = np.array([200,0,0])
        upper = np.array([255,255,255])
        someblue_mask = cv2.inRange(countour_OR, lower, upper)
        final_image = cv2.bitwise_and(countour_OR, countour_OR, mask=someblue_mask)

        lower_grey = np.array([50,50,50])
        grey_mask = cv2.inRange(img, lower_grey, upper)
        mesh  = cv2.bitwise_and(img, img, mask=grey_mask)
        final_image = cv2.bitwise_or(mesh, final_image)
        return final_image
    return bitwiseAnd

def draw_iris_DEPRECATED(mask, img, shape, results):
    height, width, channels = shape
    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])

    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
    # turn center points into np array 
    center_left  = np.array([l_cx, l_cy], dtype=np.int32)
    center_right = np.array([r_cx, r_cy], dtype=np.int32)

    LEFT_ELIPSE  = np.append(mesh_points[LEFT_IRIS], center_left.reshape(1, -1), axis=0)
    RIGHT_ELIPSE = np.append(mesh_points[RIGHT_EYE], center_right.reshape(1, -1), axis=0)

    left_ellipse  = cv2.fitEllipse(LEFT_ELIPSE)
    right_ellipse = cv2.fitEllipse(RIGHT_ELIPSE)
    # turn center points into np array 
    #center_left = np.array([l_cx, l_cy], dtype=np.int32)
    #center_right = np.array([r_cx, r_cy], dtype=np.int32)

    iris=cv2.ellipse(img, left_ellipse, COLOR, -1)
    iris=cv2.ellipse(iris, right_ellipse, COLOR, -1)
    bitwiseAnd = cv2.bitwise_and(mask, iris)
    return bitwiseAnd

def draw_face_tesselation(img, face_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())

def draw_face_keypoints(img, face_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1,
                                                                    circle_radius=0))
def draw_iris_mesh(img, face_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
          image=img,
          landmark_list=face_landmarks,
          connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())
    
class FacialLandmarks478:
    """
    Extract 478 facial landmark points from the picture and return it in a 2-dimensional picture.
    """

    def __call__(self, pic: np.ndarray, mesh_congiguration: str = '03_pkb') -> np.ndarray:
        """
        @param pic (numpy.ndarray): Image to be converted to a facial landmark image
        with 478 points.
        @return: numpy.ndarray: Converted image.
        """
        point_image = np.zeros(pic.shape, np.uint8)
        with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks is not None and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]
                point_image = config(mesh_congiguration, point_image, face_landmarks, pic, results)
           
            return point_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


