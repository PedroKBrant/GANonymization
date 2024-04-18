import cv2
import numpy as np
import math

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
        return (axis1, axis2)
    
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

points1 = np.array([[55, 40],
                    [53, 28],
                    [31, 40],
                    [53, 41]])

points2 = np.array([[125, 40],
                    [123, 33],
                    [101, 40],
                    [123, 41]])

eyes = Eyes(points1, points2)
image = np.zeros((500, 500, 3), dtype=np.uint8)  # Creating a black image
eyes.draw_on_image(image)  # Drawing both ellipses on the image

# Show the image
cv2.imshow('Ellipses', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
