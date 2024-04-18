import cv2
import numpy as np

class Ellipse:
    def __init__(self, points):
        self.points = points
        self.centroid = self.calculate_centroid()
        self.ellipse_params = self.fit_ellipse()

    def __str__(self):
        attributes = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"Ellipse({attributes})"
    
    def calculate_centroid(self):
        if len(self.points) != 4:
            raise ValueError("Four points are required to calculate the centroid.")

        mean_x = sum(point[0] for point in self.points) / len(self.points)
        mean_y = sum(point[1] for point in self.points) / len(self.points)

        return (mean_x, mean_y)

    def fit_ellipse(self):
        # Convert points to numpy array with float32 data type
        points_arr = np.array(self.points + [self.centroid], dtype=np.float32)  # Include centroid as the fifth point

        # Fit ellipse to the points using OpenCV's fitEllipse function
        ellipse_params = cv2.fitEllipse(points_arr)

        return ellipse_params

    def draw_on_image(self, image):
        # Ensure image is color (3 channels)
        if len(image.shape) != 3:
            raise ValueError("Image must be a color image (3 channels).")

        # Draw the ellipse on the image
        cv2.ellipse(image, self.ellipse_params, (0, 255, 0), 2)
        
        return image
class Eyes:
    def __init__(self, points1, points2):
        self.ellipse_left = Ellipse(points1)
        self.ellipse_right = Ellipse(points2)


# Example usage
points1 = [(70, 20), (30, 40), (50, 60), (70, 80)]
points2 = [(15, 25), (35, 45), (55, 65), (75, 85)]

real_sample = np.array([[75, 40],
                        [73, 38],
                        [71, 40],
                        [73, 41]])

eyes = Eyes(points1, points2)
print(eyes.ellipse_left.centroid)

# Create a blank image (1000x1000 pixels) to draw the ellipse on
image = np.zeros((500, 500, 3), dtype=np.uint8)

# Draw the ellipse on the image
image_with_ellipse = eyes.ellipse_left.draw_on_image(image)

print(eyes.ellipse_left)
# Show the image with the ellipse
cv2.imshow("Ellipse", image_with_ellipse)
cv2.waitKey(0)
cv2.destroyAllWindows()
