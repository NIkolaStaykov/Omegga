import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize(image):
    (h, w) = image.shape[:2]
    hw_ratio = h/w
    pixels = 518400
    h, w = (int(np.sqrt(hw_ratio*pixels)), int(np.sqrt(pixels/hw_ratio)))
    return cv2.resize(image, (w, h))


def get_ellipses(image, debug = False):
    shape = image.shape[:2]
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
    segmented_image = segmented_image.reshape(shape)

    el_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    segmented_image = cv2.erode(segmented_image, el_1, iterations=3)
    if debug:
        cv2.imshow("segmented", segmented_image)
    # Canny Edge Detection
    edges = cv2.Canny(image=segmented_image, threshold1=20, threshold2=50)  # Canny Edge Detection

    el_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Display Canny Edge Detection Image
    edges = cv2.dilate(edges, el_2, iterations=1)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        cv2.imshow("Edges", edges)

    big_contours_ind = []
    ellipses = []
    for i, contour in enumerate(contours):
        if cv2.arcLength(contour, True) > 70 and cv2.contourArea(contour) != 0:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h > 10000:
                big_contours_ind.append(i)

    canvas = np.zeros(shape)
    for ind in big_contours_ind:
        if hierarchy[0][ind][3] not in big_contours_ind:
            ellipse = cv2.fitEllipse(contours[ind])
            ellipses.append(ellipse)
            cv2.ellipse(canvas, ellipse, (255, 255, 0))
        cv2.imshow("Ellipses", canvas)

    if debug:
        used_contours = []
        canvas = np.zeros(shape)
        for ind in big_contours_ind:
            used_contours.append(contours[ind])
        cv2.drawContours(canvas, used_contours, -1, (0, 0, 255), 1)
        cv2.imshow("Edges", edges)
        cv2.imshow("Contours", canvas)

    return ellipses


def in_ellipse(point_coords, ellipse):
    (px, py) = point_coords
    elx, ely = ellipse[0]
    ell_radii = ellipse[1]
    ell_angle = ellipse[2]
    return ((np.cos(ell_angle)*(px - elx) + np.sin(ell_angle)*(py-ely))**2)/(ell_radii[0]**2) +\
           ((np.sin(ell_angle)*(px - elx) - np.cos(ell_angle)*(py-ely))**2)/(ell_radii[1]**2) < 1


def fit_ellipse():
    pass


def get_direction():
    pass


if __name__ == "__main__":
    vid = cv2.VideoCapture(1)
    while True:
        ret, img = vid.read()
        cv2.imshow("og", img)
        ellipses = get_ellipses(img, True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
