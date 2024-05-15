import cv2
import matplotlib.pyplot as plt


def show_img(original_img, cropped_img, img_name):
    """
    The function accepts the original image with its cropped variant in an array like format. The additional argument
    :param img_name is used to specify the title of the plot.
    :param original_img: array representing the original image in pixels.
    :param cropped_img:  array representing the cropped image in pixels.
    :param img_name: the image name that will be shown in the generated plot.
    :return:
        NoneType
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))

    ax[0].set_title(f"Original Image - {img_name}")
    ax[0].imshow(original_img, cmap="gray")

    ax[1].set_title(f"Cropped Image - {img_name}")
    ax[1].imshow(cropped_img, cmap="gray")


def crop_image(img, height, width, verbose=0, length=-1):
    """
    This function crops the image based on the dimensions passed as input parameters.
    :param img: (array) representing the original image in pixels.
    :param height: (int) the target height of the image.
    :param width: (int) the target width of the image.
    :param verbose: (bool) specify if the logs should be displayed.
    :param length: (int) if specified and there's inconsistency with multiple images, this parameter resolves the marginal
    error.
    :return:
    (tuple) in the following order (squared_crop, x_min, x_max, y_min, y_max). The squared_crop is the cropped square
    image with the correct height and width, the variables x_min, x_max, y_min, y_max can be used to form the (x, y)
    coordinates of each edge of the cropped image.
    """

    def calculate_starting_position(img, type, x, y, value, to_compare):
        """
        This function calculates the starting position of each image with an abstraction from the height and width
        differences. It iterates through the pixels from the top/bottom/left/right middle position until it reaches a
         pixel different than the one specified as an input parameter.
        :param img:  array representing the original image in pixels.
        :param type: [h or w] represents if the function should calculate the starting point for the width or height.
        :param x: the staring x coordinate from which the starting point should be calculated.
        :param y: the staring y coordinate from which the starting point should be calculated.
        :param value: the step used to determine the next pixel value.
        :param to_compare: the value that each pixel should be compared to.
        :return:
            the pixel coordinate (either x or y) from the pixel that is different from the one specified as the
            input parameter.
        """

        def compare_channels(channel_arr):
            """
            Accepts the bool result of comparing two (r,g,b) channel arrays and compares the sum to the length.
            If they are equal then the pixel values are equal as well.
            :param channel_arr: bool result from comparing two (r,g,b) channel arrays.
            :return:
             bool. True if the two channels are equal to each other.
            """
            return sum(channel_arr) == len(channel_arr)

        while compare_channels(img[x, y] == to_compare):
            if type == "w":
                x += value
            else:
                y += value
        return x if type == "w" else y

    half_height = int(img.shape[0] / 2)
    half_width = int(img.shape[1] / 2)
    if verbose:
        print(f"The image has a width of: {width} / 2 - {half_width}, a height of: {height} / 2 - {half_height}")

    thresh = 127
    (_, img_preprocessed) = cv2.threshold(img.copy(), thresh, 255, cv2.THRESH_BINARY)

    upper_middle = img_preprocessed[0][half_width]
    low_middle = img_preprocessed[-2][half_width]

    left_middle = img_preprocessed[half_height][0]
    right_middle = img_preprocessed[half_height][-100]

    y_min = calculate_starting_position(img_preprocessed, type="w", x=0, y=half_width, value=1, to_compare=upper_middle)
    y_max = calculate_starting_position(img_preprocessed, type="w", x=height - 2, y=half_width, value=-1,
                                        to_compare=low_middle)

    x_min = calculate_starting_position(img_preprocessed, type="h", x=half_height, y=0, value=1, to_compare=left_middle)
    x_max = calculate_starting_position(img_preprocessed, type="h", x=half_height, y=width - 100, value=-1,
                                        to_compare=right_middle)

    crop = img[y_min:y_max, x_min:x_max]

    if length == -1:
        length = abs(x_max - x_min)

    up_points = (length, length)
    squared_crop = cv2.resize(crop, up_points, interpolation=cv2.INTER_LINEAR)
    return squared_crop, x_min, x_max, y_min, y_max
