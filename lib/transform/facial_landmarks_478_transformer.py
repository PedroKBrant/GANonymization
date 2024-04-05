import cv2
import mediapipe as mp
import numpy as np

circleDrawingSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

LEFT_IRIS = [474,475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def face_contour():
    mp.solutions.drawing_utils.draw_landmarks(
        image=point_image,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    
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

                # # Draw face landmarks
                # mp.solutions.drawing_utils.draw_landmarks(
                #     image=point_image,
                #     landmark_list=face_landmarks,
                #     landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1,
                #                                                                 circle_radius=0))
                mp.solutions.drawing_utils.draw_landmarks(
                    image=point_image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())
                
                mp.solutions.drawing_utils.draw_landmarks(
                    image=point_image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())
                
                # Draw iris landmarks at the middle of the eye
                if results.multi_face_landmarks[0].landmark:
                    height, width, channels = pic.shape
                    mesh_points=np.array([np.multiply([p.x, p.y], [width, height]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                    (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                    (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                    # turn center points into np array 
                    center_left = np.array([l_cx, l_cy], dtype=np.int32)
                    center_right = np.array([r_cx, r_cy], dtype=np.int32)

                    cv2.circle(point_image, center_left, int(l_radius), (255,0,255), 1, cv2.LINE_AA)
                    cv2.circle(point_image, center_right, int(r_radius), (255,0,255), 1, cv2.LINE_AA)
                                  
                    # left_eye_top_index = 468
                    # right_eye_top_index = 473

                    # left_eye_top = face_landmarks.landmark[left_eye_top_index]
                    # right_eye_top = face_landmarks.landmark[right_eye_top_index]
                    # left_eye_top_x = int(left_eye_top.x * pic.shape[1])
                    # left_eye_top_y = int(left_eye_top.y * pic.shape[0])
                    # right_eye_top_x = int(right_eye_top.x * pic.shape[1])
                    # right_eye_top_y = int(right_eye_top.y * pic.shape[0])

                    # # Draw iris landmarks at the middle of the eye
                    # cv2.circle(point_image, (left_eye_top_x, left_eye_top_y), 10, (255,0,0), -1)
                    # cv2.circle(point_image, (right_eye_top_x, right_eye_top_y), 10, (255,0,0), -1)
 
            return point_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"


'''
                # mp.solutions.drawing_utils.draw_landmarks(
                #     image=point_image,
                #     landmark_list=face_landmarks,
                #     connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp.solutions.drawing_styles
                #     .get_default_face_mesh_tesselation_style())
                
                mp.solutions.drawing_utils.draw_landmarks(
                    image=point_image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())

                # mp.solutions.drawing_utils.draw_landmarks(
                #     image=point_image,
                #     landmark_list=face_landmarks,
                #     connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                #     landmark_drawing_spec=None,
                #     connection_drawing_spec=mp.solutions.drawing_styles
                #     .get_default_face_mesh_iris_connections_style())
'''
