import cv2
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt


def get_stats(image, object_size):
    """
    Turns an image into a binary format to remove the objects (noise) with a smaller size than the specified.
    :param image: (array) representing the original image in pixels.
    :param object_size: (int) the minimum size of the noise reduction.
    :return:
        (retval, labels, stats, centroids)
    """
    temp = image.copy()
    binarized = np.where(temp.copy() > 0.1, 1, 0)
    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=object_size, connectivity=2).astype(
        int)
    return cv2.connectedComponentsWithStats(np.asarray(processed, dtype="uint8"))


def segment(img, min_size, verbose=0, title=""):
    """
    This function encapsulates all the steps required to manually segment the plants using traditional Computer Vision
    (CV) only. The steps include: grayscale, blur, division, thresholding, then  clearing the noise (the small objects).
    :param img: (array) representing the original image in pixels.
    :param min_size: (int) the minimum size of the noise reduction.
    :param verbose: (bool) this parameter specified if the function should work in production or debug mode, therefore if it
    should log its steps or just return the desired output.
    :param title: (str) The title of the image that will be displayed if the function works in debug mode.
    :return:
        The segmented representation of the image passed as an input parameter.
    """
    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=53, sigmaY=53)
    divide = cv2.divide(gray, blur, scale=255)
    thresholded = cv2.threshold(divide.copy(), 240, 255, cv2.THRESH_BINARY_INV)[1]

    binarized = np.where(thresholded.copy() > 0.1, 1, 0)
    processed = morphology.remove_small_objects(binarized.astype(bool), min_size=min_size, connectivity=2).astype(int)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(np.asarray(processed, dtype="uint8"))

    temp_bg = labels.copy()
    width = temp_bg.shape[0]
    height = temp_bg.shape[1]
    temp_bg[temp_bg == temp_bg[0, 0]] = 0
    temp_bg[temp_bg == temp_bg[width - 1, height - 1]] = 0

    temp_bg[temp_bg == temp_bg[0, height - 1]] = 0
    temp_bg[temp_bg == temp_bg[width - 1, 0]] = 0

    tempp = temp_bg.copy()
    for i in range(1, len(stats)):
        if 0 not in stats[i] and width not in stats[i] and height not in stats[i]:
            x, y, w, h, _ = stats[i]
            tempp = cv2.rectangle(tempp, (x, y), (x + w, y + h), 2, 5)

    if verbose:
        fig, ax = plt.subplots(3, 3, figsize=(10, 10))

        fig.suptitle(title)

        ax[0, 0].set_title('original')
        ax[0, 0].imshow(gray, cmap="gray")
        ax[0, 1].set_title('blurred')
        ax[0, 1].imshow(blur, cmap="gray")
        ax[0, 2].set_title('divided')
        ax[0, 2].imshow(divide, cmap="gray")

        ax[1, 0].set_title('threshold')
        ax[1, 0].imshow(thresholded, cmap="gray")
        ax[1, 1].set_title('binary')
        ax[1, 1].imshow(binarized, cmap="gray")
        ax[1, 2].set_title('morphology')
        ax[1, 2].imshow(processed, cmap="gray")

        ax[2, 0].set_title('labels')
        ax[2, 0].imshow(labels, cmap="gray")
        ax[2, 1].set_title('border fix')
        ax[2, 1].imshow(temp_bg, cmap="gray")
        ax[2, 2].set_title('bounding box')
        ax[2, 2].imshow(tempp, cmap="gray")

        # plt.show()
        return temp_bg
    else:
        return temp_bg
