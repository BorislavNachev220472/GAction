import numpy as np
import cv2
import os

def create_folder(folder_name: str) -> None:
    """
    Create a folder if it doesn't exist.

    Author: Vlad Matache 224108@buas.nl
    :param folder_name: Name of the folder to create.
    """
    import os

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created successfully.")
    else:
        print(f"Folder '{folder_name}' already exists.")

def check_input_type(input_data: any) -> str:
    """
    Check the type of the input data.

    Author: Vlad Matache, 224108@buas.nl
    :param input_data: Input data to check.
    """
    if type(input_data) is str:
        return "str"
    elif type(input_data) is np.ndarray:
        return "np.ndarray"
    else:
        return "unspported"
    
def load_image(image_path: str, verbose: bool = False) -> np.ndarray:
    """
    Load an image from a given path.

    Author: Vlad Matache, 224108@buas.nl
    :param image_path: Path of the image to load.
    :param verbose: Print additional information.
    :return: Loaded image.
    """
    if check_input_type(image_path) != "str":
        raise ValueError(f"Unsupported input type: {check_input_type(image_path)}")
    else:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at: {image_path}")
        elif len(image.shape) != 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        if verbose:
            print(f"Loaded image with shape: {image.shape}")
        return image

def load_images_from_folder(folder_path: str, verbose: bool = False) -> tuple[list,list]:
    """
    Load images from a given folder.

    Author: Vlad Matache, 224108@buas.nl
    :param folder_path: Path of the folder containing the images.
    :param verbose: Print additional information.
    :return: List of loaded images.
    """
    if check_input_type(folder_path) != "str":
        raise ValueError(f"Unsupported input type: {check_input_type(folder_path)}")
    else:
        images = []
        filenames = []
        for filename in os.listdir(folder_path):
            image = load_image(os.path.join(folder_path, filename), verbose)
            images.append(image)
            filenames.append(filename)
        if verbose:
            print(f"Loaded {len(images)} images from folder: {folder_path}")
        return images, filenames