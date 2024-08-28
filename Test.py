import numpy as np
import cv2, joblib
from imutils.object_detection import non_max_suppression
import imutils
from skimage.feature import hog
from skimage import color
from skimage.transform import pyramid_gaussian


def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield x, y, image[y: y + window_size[1], x: x + window_size[0]]


image = cv2.imread('C:/Users/1/Desktop/UNI/7/computer vision/project/myproj/DATAIMAGE/negative/train/4_6.jpg')
image = cv2.resize(image, (64, 128))
size = (64, 128)
step_size = (9, 9)
downscale = 1.25
# List to store the detections
detections = []
# The current scale of the image
scale = 0
model = joblib.load('models/models.dat')
for im_scaled in pyramid_gaussian(image, downscale=downscale):
    # The list contains detections at the current scale
    if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
        break
    for (x, y, window) in sliding_window(im_scaled, size, step_size):
        if window.shape[0] != size[1] or window.shape[1] != size[0]:
            continue
        window = color.rgb2gray(window)

        fd = hog(window, orientations=9, pixels_per_cell=(8, 8), visualize=False, cells_per_block=(3, 3))
        fd = fd.reshape(1, -1)
        pred = model.predict(fd)
        if pred == 1:

            if model.decision_function(fd) > 0.5:
                detections.append(
                    (int(x * (downscale ** scale)), int(y * (downscale ** scale)), model.decision_function(fd),
                     int(size[0] * (downscale ** scale)),
                     int(size[1] * (downscale ** scale))))

    scale += 1
clone = image.copy()
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
sc = [score[0] for (x, y, score, w, h) in detections]
print("detect score: ", sc)
sc = np.array(sc)
pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)
for (x1, y1, x2, y2) in pick:
    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(clone, 'Person', (x1 - 2, y1 - 2), 1, 0.75, (121, 12, 34), 1)
cv2.imshow('Person Detection', clone)
cv2.waitKey(0)
cv2.destroyAllWindows()
