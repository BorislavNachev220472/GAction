import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from patchify import patchify, unpatchify

import tensorflow as tf
import keras.backend as K
from keras.models import load_model

from skimage.morphology import skeletonize
from skan import Skeleton, summarize
from skan.csr import skeleton_to_csgraph
from skan import draw
import skan
from skimage.morphology import skeletonize
import networkx as nx

def find_start_end(array: np.ndarray) -> list[int]:
    '''
    Takes is an image as array and return start and end coordinates of the petri dish

    Author: Fedya Chursin fedorchursinsk@gmail.com

    :param array: Image as a numpy array.
    :return: Start and end coordinates of the petri dish.
    '''
    start_point = np.argmin(array)
    end_point = np.argmax(array)

    return [start_point, end_point]



def derivative(image: np.ndarray, axis: int) -> np.ndarray:
    '''
    Takes in an image as an array and return average derivative value on x or y-axis

    Author: Fedya Chursin fedorchursinsk@gmail.com

    :param arr: Image as an array.
    :param axis: Integer indicating the axis along which to compute the derivative.
    :return: Average derivative value on x or y-axis.
    '''
    if axis:
        derivatives = np.gradient(image, axis=1)
        average_derivatives_vertical = np.mean(derivatives, axis=0)
        return average_derivatives_vertical

    else:
        derivatives = np.gradient(image, axis=0)
        average_derivatives_vertical = np.mean(derivatives, axis=1)
        return average_derivatives_vertical


def roi_extraction_coords_direct(image : np.ndarray) -> list[int,int,int,int]: 
    '''
    Function takes in a path to an image and returns coordinates of the petri dish in it by finding average derivatives values, start and end coordinates of the petri dish by looking at the change of the derivative. These coordinates are used to crop masks
    
    Author: Fedya Chursin fedorchursinsk@gmail.com

    :param path: Path to an image.
    :return: Coordinates of the petri dish to crop masks.
    '''
    image = image[0:-1, 0:4000]

    average_derivatives_horizontal = derivative(image, 1)
    average_derivatives_vertical = derivative(image, 0)

    y = find_start_end(average_derivatives_vertical)
    x = find_start_end(average_derivatives_horizontal)

    height = x[0] - x[1]
    width = y[0] - y[1]
    return [y[1] + 185 , y[1] + height , x[1] + 15, x[1] + width - 20]


def set_outside_pixels_to_zero(image: np.ndarray, min_x: int, max_x: int, min_y: int, max_y: int) -> np.ndarray:
    """
    Set outside pixels of the image to a specified value.

    Author: Vlad Matache 224108@buas.nl

    
    :param image: Input image. (np.ndarray)
    :param min_x: Minimum x-coordinate. 
    :param max_x: Maximum x-coordinate. 
    :param min_y: Minimum y-coordinate. 
    :param max_y: Maximum y-coordinate. 
    :return: Processed image. (np.ndarray)
    """
    roi_extracted_image = image
    roi_extracted_image[0:min_x] = 0
    roi_extracted_image[max_x:] = 0
    roi_extracted_image[:, 0: min_y] = 0
    roi_extracted_image[:, max_y:] = 0
    
    return roi_extracted_image

def padder(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Pad the given image to match the patch size.

    Author: Vlad Matache 224108@buas.nl

    
    :param image: Input image. 
    :param patch_size: Size of the patch. 
    :return: Padded image. 
    """
    h = image.shape[0]
    w = image.shape[1]
    height_padding = ((h // patch_size) + 1) * patch_size - h
    width_padding = ((w // patch_size) + 1) * patch_size - w

    top_padding = int(height_padding/2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding/2)
    right_padding = width_padding - left_padding
    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, cv2.BORDER_REPLICATE)
    
    return padded_image


