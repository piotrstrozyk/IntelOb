import cv2
import numpy as np


def grayscale_avg(image):
    gray = np.mean(image, axis=2)
    return gray.astype(np.uint8)

def grayscale_weighted(image):
    weights = [0.299, 0.587, 0.114]
    gray = np.dot(image[..., :3], weights)
    return gray.astype(np.uint8)

image = cv2.imread('2notes/Bird-Friendly-City.jpg')
image2 = cv2.imread('2notes/bocian.jpg')

# Konwersja za pomocą średniej
gray_avg = grayscale_avg(image)
gray_avg2 = grayscale_avg(image2)

# Konwersja za pomocą wag kanałów
gray_weighted = grayscale_weighted(image)
gray_weighted2 = grayscale_weighted(image2)

cv2.imwrite("gray_avg.jpg", gray_avg)
cv2.imwrite("gray_weighted.jpg", gray_weighted)
cv2.imshow("Original", image)
cv2.imshow("Grayscale (Average)", gray_avg)
cv2.imshow("Grayscale (Weighted)", gray_weighted)
cv2.imwrite("gray_avg2.jpg", gray_avg2)
cv2.imwrite("gray_weighted2.jpg", gray_weighted2)
cv2.imshow("Original2", image2)
cv2.imshow("Grayscale (Average)2", gray_avg2)
cv2.imshow("Grayscale (Weighted)2", gray_weighted2)
cv2.waitKey(0)
cv2.destroyAllWindows()