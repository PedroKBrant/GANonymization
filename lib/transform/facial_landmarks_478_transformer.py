import cv2
import mediapipe as mp
import numpy as np

circleDrawingSpec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

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
                
                mp.solutions.drawing_utils.draw_landmarks(
                    image=point_image,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())
            
        return point_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
