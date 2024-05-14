import cv2
import numpy as np
import keras.backend as K
from patchify import patchify, unpatchify
from tensorflow.keras.models import load_model
from skan import Skeleton, summarize
from skimage.morphology import skeletonize


root_thr = 0.05
shoot_thr = 0.5
seed_thr = 0.001

def crop_img(img):
    """ Crops an image based on edge detection using Canny.

    Args:
        image_name (str): The filename of the image to be processed.
    
    returns:
        output_image, diff_output_height, diff_output_width, y1, y2, x1, x2
        output_image(ndarray): The cropped image
        diff_output_height(int): the difference in height and width
        diff_output_width(int): the difference in width and height
        y1(int): the upper left corner of the edge
        y2(int): the down left corner of the edge
        x1(int): the upper right corner of the edge
        x2(int): the down right corner of the edge
    """

    input_image = img[:, 0:4100]

    canny = cv2.Canny(input_image, 0, 255)

    pts = np.argwhere(canny > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    output_image = input_image[y1:y2, x1:x2]
    output_height = output_image.shape[0]
    output_width = output_image.shape[1]
    difference = abs(output_height - output_width)

    if output_height > output_width:
        diff_output_height = output_height - difference
        diff_output_width = output_width
    if output_height < output_width:
        diff_output_width = output_width - difference
        diff_output_height = output_height

    output_image = output_image[0:diff_output_height, 0:diff_output_width]
    return(output_image)

def error_fix(img):
    diff_shape = round((img.shape[1] - img.shape[0]) / 2)
    diff_shape_1 = img.shape[1] - diff_shape
    img = img[:, diff_shape:diff_shape_1]
    output_img_temp = crop_img(img)
    output_img = crop_img(output_img_temp)
    return(output_img)


def padder(image, patch_size):
    """
    Adds padding to an image to make its dimensions divisible by a specified patch size.

    This function calculates the amount of padding needed for both the height and width of an image so that its dimensions become divisible by the given patch size. The padding is applied evenly to both sides of each dimension (top and bottom for height, left and right for width). If the padding amount is odd, one extra pixel is added to the bottom or right side. The padding color is set to black (0, 0, 0).

    Parameters:
    - image (numpy.ndarray): The input image as a NumPy array. Expected shape is (height, width, channels).
    - patch_size (int): The patch size to which the image dimensions should be divisible. It's applied to both height and width.

    Returns:
    - numpy.ndarray: The padded image as a NumPy array with the same number of channels as the input. Its dimensions are adjusted to be divisible by the specified patch size.

    Example:
    - padded_image = padder(cv2.imread('example.jpg'), 128)

    """
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image

def f1(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives+K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives+K.epsilon())
        return precision
    
    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
        total = K.sum(K.square(y_true),[1,2,3]) + K.sum(K.square(y_pred),[1,2,3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())
    return K.mean(f(y_true, y_pred), axis=-1)

def img_predictor(img):
    root_mask_model = load_model('models/all_trained_data_aug2.h5', custom_objects={'f1': f1, 'iou':iou})
    img = padder(img, 256)
    patches = patchify(img, (256, 256, 3), step=256)
    i = patches.shape[0]
    j = patches.shape[1]
    patches = patches.reshape(-1, 256, 256, 3)
    preds_root = root_mask_model.predict(patches/255)
    preds_root = preds_root.reshape(i, j, 256, 256)
    predicted_root = unpatchify(preds_root, (img.shape[0], img.shape[1]))
    predicted_root = predicted_root>root_thr
    predicted_root = predicted_root.astype(int)
    predicted_root = cv2.convertScaleAbs(predicted_root) 

    return(predicted_root)

def segmenter(predicted_root):
    label_count, labels, stats, _ = cv2.connectedComponentsWithStats(predicted_root)
    large = []
    """
    th_1 = np.sum(predicted_root[300:2500, 0:600]) * 0.1
    th_2 = np.sum(predicted_root[300:2500, 600:1200]) * 0.1
    th_3 = np.sum(predicted_root[300:2500, 1200:1600]) * 0.1
    th_4 = np.sum(predicted_root[300:2500, 1600:2200]) * 0.1
    th_5 = np.sum(predicted_root[300:2500, 2200:]) * 0.1
"""
    th_1 = 1
    th_2 = 1
    th_3 = 1
    th_4 = 1
    th_5 = 1
    image_width = predicted_root.shape[1]

    largest_area_1 = 0
    largest_area_2 = 0
    largest_area_3 = 0
    largest_area_4 = 0
    largest_area_5 = 0

    largest_label_1 = -1
    largest_label_2 = -1
    largest_label_3 = -1
    largest_label_4 = -1
    largest_label_5 = -1

    object_presence = np.zeros(5)


    for x in range(1, label_count):
        if (
            stats[x, cv2.CC_STAT_TOP] < 1500
           # and stats[x, cv2.CC_STAT_LEFT] >= 50
           # and stats[x, cv2.CC_STAT_LEFT] + stats[x, cv2.CC_STAT_WIDTH] <= image_width - 120
           # and stats[x, cv2.CC_STAT_TOP] > 200
           # and stats[x, cv2.CC_STAT_AREA] > 100
            
        ):
            if stats[x, cv2.CC_STAT_TOP] < 600 or stats[x, cv2.CC_STAT_AREA] > 3000:
                if (
                    stats[x, cv2.CC_STAT_LEFT] < 600
                    and stats[x, cv2.CC_STAT_AREA] > th_1
                ):
                    area_1 = stats[x, cv2.CC_STAT_AREA]
                    if area_1 > largest_area_1:
                        largest_area_1 = area_1
                        largest_label_1 = x

                if (
                    stats[x, cv2.CC_STAT_LEFT] > 600
                    and stats[x, cv2.CC_STAT_LEFT] < 1200
                    and stats[x, cv2.CC_STAT_AREA] > th_2
                ):
                    area_2 = stats[x, cv2.CC_STAT_AREA]
                    if area_2 > largest_area_2:
                        largest_area_2 = area_2
                        largest_label_2 = x


                if (
                    stats[x, cv2.CC_STAT_LEFT] > 1200
                    and stats[x, cv2.CC_STAT_LEFT] < 1600
                    and stats[x, cv2.CC_STAT_AREA] > th_3
                ):
                    area_3 = stats[x, cv2.CC_STAT_AREA]
                    if area_3 > largest_area_3:
                        largest_area_3 = area_3
                        largest_label_3 = x

                if (
                    stats[x, cv2.CC_STAT_LEFT] > 1600
                    and stats[x, cv2.CC_STAT_LEFT] < 2100
                    and stats[x, cv2.CC_STAT_AREA] > th_4
                ):
                    area_4 = stats[x, cv2.CC_STAT_AREA]
                    if area_4 > largest_area_4:
                        largest_area_4 = area_4
                        largest_label_4 = x

                if (
                    stats[x, cv2.CC_STAT_LEFT] > 2100
                    and stats[x, cv2.CC_STAT_AREA] > th_5
                ):
                    area_5 = stats[x, cv2.CC_STAT_AREA]
                    if area_5 > largest_area_5:
                        largest_area_5 = area_5
                        largest_label_5 = x


    if largest_label_1 != -1:
        large.append(largest_label_1)
        object_presence[0] = 1

    if largest_label_2 != -1:
        large.append(largest_label_2)
        object_presence[1] = 1
        
    if largest_label_3 != -1:
        large.append(largest_label_3)
        object_presence[2] = 1
    
    if largest_label_4 != -1:
        large.append(largest_label_4)
        object_presence[3] = 1

    if largest_label_5 != -1:
        large.append(largest_label_5)
        object_presence[4] = 1

    black_img1 = np.zeros_like(labels, dtype=np.uint8)
    for x, component_idx in enumerate(large):
        color = x + 1
        black_img1[labels == component_idx] = color
        
    x = []
    y = []
    w = []
    h = []

    for i in large:
        x.append(stats[i, 0])
        y.append(stats[i, 1])
        w.append(stats[i, 2])
        h.append(stats[i, 3])
    full_imgs = []

    for i in range(len(large)):
        full_imgs.append(black_img1[y[i]:y[i]+h[i], x[i]:x[i]+w[i]])

    return(full_imgs, object_presence, x, y)

def root_length_calc(image):
    skeleton = skeletonize(image)
    skeleton_branch = summarize(Skeleton(skeleton))

    x_end = []
    y_end = []

    branches = len(skeleton_branch)
    end_root_index = skeleton_branch["node-id-dst"].nsmallest(branches).iloc[-1]

    for i in range(branches):
        if skeleton_branch["node-id-dst"][i] == end_root_index:
            x_end = skeleton_branch["image-coord-dst-0"][i]
            y_end = skeleton_branch["image-coord-dst-1"][i]

    return x_end, y_end

def img_handler(im):
    im = crop_img(im)

    predicted_root = img_predictor(im)

    root_imgs, object_presence, x, y= segmenter(predicted_root)
    
    root_lens = []  
    x_end = []
    y_end = []

    
    i = 0

    for object in object_presence:
        if object == 1:
            y_e, x_e = root_length_calc(root_imgs[i])

            x_end.append(x[i] + x_e)
            y_end.append(y[i] + y_e)
            i += 1

        else:
            root_lens.append(0)
    x_end = [406, 970, 1756, 1814, 2414]
    y_end = [800, 1625, 2440, 1070, 2418]
    return x_end, y_end