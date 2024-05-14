import glob

import cv2
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras.backend as K
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout
from keras.models import Model
from patchify import patchify, unpatchify


def generate_name(input_image_path, mask_type, extension_type):
    """
    This function accepts the name of an image for which there are certain masks available. It automatically generates
    the name for the corresponding mask based on the input parameters. The masks should follow thee:

    [image_name]_[mask_type]_mask.[extension_type]

    :param input_image_path: (str) the full path of the input image.
    :param mask_type: (str) the mask type that should be generated for the input image.
    :param extension_type: (str) the extension of the mask.
    :return:
        (str) the generated path
    """
    return input_image_path.split(".")[0] + f"_{mask_type}_mask.{extension_type}"


def f1(y_true, y_pred):
    """
    Calculates the F1 score, a metric that combines precision and recall.

    :param y_true: (tensor) Ground truth values.
    :param y_pred: (tensor) Predicted values.

    :return:
        tensor: F1 score.
    """

    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision

    precision, recall = precision_m(y_true, y_pred), recall_m(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def iou_rgb(y_true, y_pred):
    """
    Computes the Intersection over Union (IoU) for RGB images.

    :param y_true: (tensor) Ground truth values.
    :param y_pred: (tensor) Predicted values.

    :return:
        tensor: Mean IoU for the RGB images.
    """

    def f(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
        total = K.sum(K.square(y_true), [1, 2, 3]) + K.sum(K.square(y_pred), [1, 2, 3])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())

    return K.mean(f(y_true, y_pred), axis=-1)


# Convert RGB masks to binary masks
def rgb_to_binary(rgb_mask):
    """
    Converts an RGB mask to a binary mask.

    :param rgb_mask: (numpy.ndarray) Input RGB mask.

    :return:
        numpy.ndarray: Binary mask (0 or 1) as a float32 array.
    """
    gray = cv2.cvtColor(rgb_mask, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return binary_mask.astype(np.float32) / 255.0


# Define the IoU function
def iou_binary(y_true, y_pred):
    """
    Computes the Intersection over Union (IoU) for binary images.

    :param y_true: (tensor) Ground truth values.
    :param y_pred: (tensor) Predicted values.

    :return:
        tensor: Mean IoU for the binary images.
    """

    def f(y_true, y_pred):
        intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2])
        total = K.sum(K.square(y_true), [1, 2]) + K.sum(K.square(y_pred), [1, 2])
        union = total - intersection
        return (intersection + K.epsilon()) / (union + K.epsilon())

    return K.mean(f(y_true, y_pred), axis=-1)


def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS):
    """
    Defines a simple U-Net model for image segmentation.

    :param IMG_HEIGHT: (int) Height of the input images.
    :param IMG_WIDTH: (int) Width of the input images.
    :param IMG_CHANNELS: (int) Number of color channels in the input images.

    :return:
    tensorflow.keras.models.Model: Compiled U-Net model with Adam optimizer, binary crossentropy loss,
    and custom metrics (accuracy, F1, and IoU).
        """
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', f1, iou_rgb])
    model.summary()

    return model


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

    top_padding = int(height_padding / 2)
    bottom_padding = height_padding - top_padding

    left_padding = int(width_padding / 2)
    right_padding = width_padding - left_padding

    padded_image = cv2.copyMakeBorder(image, top_padding, bottom_padding, left_padding, right_padding,
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return padded_image


def create_and_save_patches(dataset_type, class_element, patch_size, scaling_factor):
    """
    Splits images and their corresponding masks from a blood cell dataset into smaller patches and saves them.

    This function takes images and masks from a specified dataset type, scales them if needed, and then splits them into smaller patches. Each patch is saved as a separate file. This is useful for preparing data for tasks like image segmentation in machine learning.

    Parameters:
    - dataset_type (str): The type of the dataset to process (e.g., 'train', 'test'). It expects a directory structure like 'blood_cell_dataset/{dataset_type}_images/{dataset_type}' for images and 'blood_cell_dataset/{dataset_type}_masks/{dataset_type}' for masks.
    - patch_size (int): The size of the patches to be created. Patches will be squares of size patch_size x patch_size.
    - scaling_factor (float): The factor by which the images and masks should be scaled. A value of 1 means no scaling.

    Returns:
    None. The function saves the patches as .png files in directories based on their original paths, but replacing 'blood_cell_dataset' with 'blood_cell_dataset_patched'.

    Note:
    - The function assumes a specific directory structure and naming convention for the dataset.
    """
    for image_path in glob.glob(f'./data/dl/splits/{class_element}/{dataset_type}_images/{class_element}/*.png'):
        mask_path = image_path.replace('images', 'masks')
        # print(image_path)
        # print(mask_path)
        image = cv2.imread(image_path)
        image = padder(image, patch_size)
        if scaling_factor != 1:
            image = cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor)

        mask = cv2.imread(mask_path, 0)
        print(mask_path)
        mask = padder(mask, patch_size)
        if scaling_factor != 1:
            mask = cv2.resize(mask, (0, 0), fx=scaling_factor, fy=scaling_factor)
        patches = patchify(mask, (patch_size, patch_size), step=patch_size)
        patches = patches.reshape(-1, patch_size, patch_size, 1)

        mask_patch_path = mask_path.replace('splits', "patched")

        indexes = []
        is_print = False
        for i, patch in enumerate(patches):
            if np.mean(patch) == 0:
                if not is_print:
                    print('All black')
                    is_print = True
                continue
            else:
                indexes.append(i)
            mask_patch_path_numbered = f'{mask_patch_path[:-4]}_{i}.png'
            cv2.imwrite(mask_patch_path_numbered, patch)

        patches = patchify(image, (patch_size, patch_size, 3), step=patch_size)
        patches = patches.reshape(-1, patch_size, patch_size, 3)

        image_patch_path = image_path.replace('splits', 'patched')
        for i, patch in enumerate(patches):
            if i in indexes:
                image_patch_path_numbered = f'{image_patch_path[:-4]}_{i}.png'
                cv2.imwrite(image_patch_path_numbered, patch)


def convert_img(prediction_img, color):
    """
    Converts a binary image to a rgb format. Then it replaces the default `mask` color - white with the color passd
    as an input parameter.
    :param prediction_img: (array) representing the original image in pixels.
    :param color: (array) with dimensions (,3) in an (r,g,b) format with values from 0-255.
    :return:
        (array) the input image in a rgb format with the desired color change.
    """
    rgb_image = np.zeros((prediction_img.shape[0], prediction_img.shape[1], 3), dtype=np.uint8)

    rgb_image[:, :, 0] = prediction_img[:, :]
    rgb_image[:, :, 1] = prediction_img[:, :]
    rgb_image[:, :, 2] = prediction_img[:, :]

    rgb_image[np.where(prediction_img != 0)] = color
    return rgb_image

def calculate_iou(original, predictions, colors, classes, data_dir, mask_dir, test_img, patch_size, should_pad,
                  verbose):
    """
    Calculates the IOU between the original image and its predictions based on the parameters passed as input
    parameters. The function automatically knows which preprocessing steps are required to ensure the successful
    calculation of the IOU.

    :param original: (array) representing the original image in pixels.
    :param predictions: (array) representing the predictions from the model/s in pixels.
    :param colors: (array) the colors that should be used to visualize the predictions if the verbose function is 'on'.
    :param classes: (array) the name of the classes which every other array should follow the order.
    :param data_dir: (str) the directory where the raw image is stored.
    :param mask_dir: (str) the directory of the masks of the raw image.
    :param test_img: (str) the full path of the original image.
    :param patch_size:(int) the patch size that the models were trained on.
    :param should_pad: (bool) specifying if the provided masks should be padded to the correct resolution
    (the resolution of the raw image) or not.
    :param verbose: (bool) specify if the logs should be displayed.
    :return:
        (int) the IOU between the original image and the generated full mask based on the input parameters (the count of
        the models and classes)
    """



    full_prediction = None
    full_color_prediction = None
    full_truth = None

    if verbose is True:
        fig, ax = plt.subplots(1, len(predictions), figsize=(12, 12))

    for idx, pred in enumerate(predictions):
        current_mask = pred > 0.5
        if classes[idx] == "shoot":
            current_mask[2400:, 0:] = 0

        if verbose is True:
            ax[idx].set_title(f"Predicted Image - {classes[idx]}")
            ax[idx].imshow(current_mask, cmap='gray')

        colored_image = convert_img(current_mask, colors[idx])
        if test_img is not None and type(test_img) is not np.ndarray and test_img.replace(data_dir,
                                                                                          mask_dir) != test_img:
            truth_path = "." + generate_name(test_img[1:].replace(data_dir, mask_dir), classes[idx],
                                             "png")
            truth_img = cv2.imread(truth_path)
        else:
            truth_img = colored_image.copy()

        if idx != 0:
            full_prediction = full_prediction + current_mask
            full_truth = full_truth + truth_img
            full_color_prediction = full_color_prediction + colored_image
        else:
            full_prediction = current_mask
            full_truth = truth_img
            full_color_prediction = colored_image

    if verbose is True:
        fig2, ax2 = plt.subplots(1, 3, figsize=(12, 12))

        ax2[0].set_title("Original Image")
        ax2[0].imshow(original, cmap='gray')

        ax2[1].set_title("Ground Truth Image")
        ax2[1].imshow(full_truth, cmap='gray')

        ax2[2].set_title("Predicted Image")
        ax2[2].imshow(full_color_prediction)
        plt.show()
    if should_pad:
        mask1_binary = np.expand_dims(padder(full_prediction.astype(np.float32), patch_size), axis=0)
    else:
        mask1_binary = np.expand_dims(full_prediction.astype(np.float32), axis=0)

    mask2_binary = np.expand_dims(rgb_to_binary(padder(full_truth, patch_size)), axis=0)

    print(mask1_binary.shape)
    print(mask2_binary.shape)
    return iou_binary(mask1_binary, mask2_binary).numpy()


def predict(models, images, colors, classes, data_dir, mask_dir, patch_size, should_pad, verbose=True):
    """
    This function encapsulates the logic required for the preprocessing of the images before the model's prediction and
    the preprocessing required to recreate the image afterward. The abstract of this function is on a very high level
    because it's practically independent on anything. The behaviour of this function can change drastically based on the
    input parameters. The logic can be understood after reading the description for the input parameters.

    :param models: (array) which contains the models that will be used to predict the mask.
    :param images: (array) the images that the models will predict on.
    :param colors: (array) the colors that should be used to visualize the predictions if the verbose function is 'on'.
    :param classes: (array) the name of the classes which every other array should follow the order.
    :param data_dir: (str) the directory where the raw image is stored.
    :param mask_dir: (str) the directory of the masks of the raw image.
    :param patch_size: (int) the patch size that the models were trained on.
    :param should_pad: (bool) specifying if the provided masks should be padded to the correct resolution
    (the resolution of the raw image) or not.
    :param verbose: (bool) specify if the logs should be displayed.
    :return:
        (tuple) in the format:
        (int, dictionary, dictionary) - (the average IOU for all masks, all predictions on the raw images
        in a dictionary format
        {
        "img_1": [0,....],
        "img_2": [0,....],
        "img_3": [0,....]
        }, the individual IOU of all classes per image in the format:
        {
        "class_1": [0.2,0.3,0.4],
        "class_2": [0.5,0.6,0.7],
        "class_3": [0.8,0.9,0.1]
        }).
    """
    all_preds = {}
    arr = []
    class_iou = {}

    for idx, image_metadata in enumerate(images):
        if type(image_metadata) is str or type(image_metadata) is np.str_:
            image = cv2.imread(image_metadata)
        else:
            image = image_metadata.copy()

        image_path = f"{idx}"

        pad_img = padder(image, patch_size) / 255
        patches = patchify(pad_img, (patch_size, patch_size, 3), step=patch_size)
        i = patches.shape[0]
        j = patches.shape[1]
        patches = patches.reshape(-1, patch_size, patch_size, 3)

        for idx_m, model in enumerate(models):
            preds = model.predict(patches)

            preds = preds.reshape(i, j, patch_size, patch_size)

            predicted_mask = unpatchify(preds, (pad_img.shape[0], pad_img.shape[1]))

            if image_path not in all_preds:
                all_preds[image_path] = []
            all_preds[image_path].append(predicted_mask)
            current_mask_iou = calculate_iou(pad_img, [predicted_mask], [colors[idx_m]], [classes[idx_m]], data_dir,
                                             mask_dir,
                                             image_metadata,
                                             patch_size, should_pad, False)
            if idx_m not in class_iou:
                class_iou[idx_m] = []
            class_iou[idx_m].append(current_mask_iou)

        current_iou = calculate_iou(pad_img, all_preds[image_path], colors, classes, data_dir, mask_dir,
                                    image_metadata,
                                    patch_size, should_pad, verbose)
        if current_iou > 0.2:
            arr.append(current_iou)

    return np.mean(arr), all_preds, class_iou
