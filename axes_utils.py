import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize(image):
    (h, w) = image.shape[:2]
    hw_ratio = h/w
    pixels = 518400
    h, w = (int(np.sqrt(hw_ratio*pixels)), int(np.sqrt(pixels/hw_ratio)))
    return cv2.resize(image, (w, h))


def get_ellipses(image):
    shape = image.shape

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # number of clusters (K)
    k = 2
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)

    labels = labels.flatten()
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(image.shape)

    # Canny Edge Detection
    edges = cv2.Canny(image=segmented_image, threshold1=20, threshold2=50)  # Canny Edge Detection

    # Display Canny Edge Detection Image
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, el, iterations=1)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    big_contours_ind = []
    ellipses = []

    canvas = np.zeros(shape, dtype='uint8')
    for i, contour in enumerate(contours):
        if cv2.arcLength(contour, True) > 70 and cv2.contourArea(contour) != 0:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > 15000:
                big_contours_ind.append(i)

    for ind in big_contours_ind:
        if hierarchy[0][ind][3] not in big_contours_ind:
            ellipse = cv2.fitEllipse(contours[ind])
            ellipses.append(ellipse)
            cv2.ellipse(canvas, ellipse, (0, 0, 255))

    cv2.imshow("Ellipses", canvas)
    cv2.waitKey(0)

    return ellipses


def fit_ellipse():
    pass


def get_direction():
    pass


if __name__ == "__main__":
    im = cv2.imread("black_bg_1.jpg")
    ellipses = get_ellipses(im)
