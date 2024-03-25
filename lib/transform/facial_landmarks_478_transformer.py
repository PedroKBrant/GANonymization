"""
Created by Fabio Hellmann.
"""

import cv2
import mediapipe
import numpy as np
from mediapipe.python.solutions.drawing_utils import DrawingSpec, WHITE_COLOR

RED_COLOR = (0, 0, 255)  # Red color for the background
WHITE_COLOR = (255, 255, 255)

class FacialLandmarks478:
    """
    Extract 468 facial landmark points from the picture and return it in a 2-dimensional picture.
    """

    def __call__(self, pic: np.ndarray) -> np.ndarray:
        """
        @param pic (numpy.ndarray): Image to be converted to a facial landmark image
        with 468 points.
        @return: numpy.ndarray: Converted image.
        """
        point_image = np.zeros(pic.shape, np.uint8)
        with mediapipe.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
        
            if results.multi_face_landmarks is not None and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]

                # Draw face landmarks
                mediapipe.solutions.drawing_utils.draw_landmarks(
                    image=point_image,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=mediapipe.solutions.drawing_utils.DrawingSpec(color=WHITE_COLOR, thickness=1,
                                                                                circle_radius=0))
                

                # Draw iris landmarks at the middle of the eye
                if results.multi_face_landmarks[0].landmark:
                    left_eye_top_index = 159
                    left_eye_bottom_index = 145
                    right_eye_top_index = 386
                    right_eye_bottom_index = 374
                    
                    left_eye_top = face_landmarks.landmark[left_eye_top_index]
                    left_eye_bottom = face_landmarks.landmark[left_eye_bottom_index]
                    right_eye_top = face_landmarks.landmark[right_eye_top_index]
                    right_eye_bottom = face_landmarks.landmark[right_eye_bottom_index]
                    
                    # Calculate middle points
                    left_eye_middle_x = int((left_eye_top.x + left_eye_bottom.x) * pic.shape[1] / 2)
                    left_eye_middle_y = int((left_eye_top.y + left_eye_bottom.y) * pic.shape[0] / 2)
                    right_eye_middle_x = int((right_eye_top.x + right_eye_bottom.x) * pic.shape[1] / 2)
                    right_eye_middle_y = int((right_eye_top.y + right_eye_bottom.y) * pic.shape[0] / 2)
                    
                    # Draw iris landmarks at the middle of the eye
                    cv2.circle(point_image, (left_eye_middle_x, left_eye_middle_y), 2, RED_COLOR, -1)
                    cv2.circle(point_image, (right_eye_middle_x, right_eye_middle_y), 2, RED_COLOR, -1)
        return point_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
