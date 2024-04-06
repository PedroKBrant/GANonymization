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

def draw_eye_region(img, shape, results, fill=True):
    height, width, channels = shape
    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])
    img_copy = img.copy()
    if fill:
        mask =cv2.fillPoly(img, [mesh_points[LEFT_EYE]], COLOR)
        mask = cv2.fillPoly(img, [mesh_points[RIGHT_EYE]], COLOR)
        return draw_iris(mask, img_copy, shape, results)
    else:
        cv2.polylines(img, [mesh_points[LEFT_EYE]], True, COLOR, 1, cv2.LINE_AA)
        cv2.polylines(img, [mesh_points[RIGHT_EYE]], True, COLOR, 1, cv2.LINE_AA)
    cv2.polylines(img, [mesh_points[INTERNAL_LIPS]], True, COLOR, 1, cv2.LINE_AA)

def draw_face_contour(img, face_landmarks):
    mp.solutions.drawing_utils.draw_landmarks(
        image=img,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())

def draw_iris(mask, img, shape, results):
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
class FacialLandmarks478:
    """
    Extract 478 facial landmark points from the picture and return it in a 2-dimensional picture.
    """

    def __call__(self, pic: np.ndarray) -> np.ndarray:
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

                #draw_face_keypoints(point_image, face_landmarks)

                draw_face_tesselation(point_image, face_landmarks)
                
                #draw_face_contour(point_image, face_landmarks)
                point_image = draw_eye_region(point_image, pic.shape, results)
                #draw_iris(point_image, pic.shape, results)

                            
            return point_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


