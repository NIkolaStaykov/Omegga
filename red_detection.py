import cv2
import numpy as np
from axes_utils import get_ellipses, in_ellipse

# define a video capture object
def score (blue, red, green):
    maximum = max(green, blue)
    if maximum>=red:
        return 0
    else:
        sc = red - maximum
        return sc



vid = cv2.VideoCapture(1)

while (True):
    ret, img = vid.read()

    cv2.imshow('frame', img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1

    # set my output img to zero everywhere except my mask
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0

    # change it with your absolute path for the image
    ret, thresh = cv2.threshold(mask, 200, 255,
                               cv2.THRESH_BINARY_INV)
    cv2.imwrite("thresh.png", thresh)

    contours, hierarchies = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(thresh.shape[:2], dtype='uint8')

    cv2.drawContours(blank, contours, -1,
                     (255, 0, 0), 1)

    cv2.imwrite("Contours.png", blank)

    cx = cy = 0
    for contour in contours:
        M = cv2.moments(contour)
        if cv2.arcLength(contour, True) > 35 and cv2.contourArea(contour) != 0:
            x, y, w, h = cv2.boundingRect(contour)
            if w*h < 30000 and w*h > 250 and w/h < 1.5 and h/w < 1.5:
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.drawContours(output_img, [contour], -1, (0, 255, 0), 2)
                    cv2.circle(output_img, (cx, cy), 7, (0, 0, 255), -1)
                    cv2.putText(output_img, "center", (cx - 20, cy - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imwrite("image.png", output_img)

    ellipses = get_ellipses(img, True)
    for ell in ellipses:
        if in_ellipse((cx, cy), ell):
            elx, ely = int(ell[0][0]), int(ell[0][1])
            cv2.circle(output_img, (elx, ely), 7, (128, 255, 255), -1)
            cv2.ellipse(output_img, ell, (0, 255, 255))

    cv2.imshow('video gray', output_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()