class Config(object):
    class_colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0)]

    split = ['train_images', 'train_masks', 'val_images', 'val_masks']

    classes = ["root", "seed", "shoot", "occluded_root"]

    splits_dir = './data/dl/splits'
    patch_dir = './data/dl/patched'

    patch_size = 256
    scaling_factor = 1

    SEED = 42
    TRAIN_SIZE = 0.9
    VALID_SIZE = 0.1

    preprocessed_location = "./data/dl/cropped/train_val_images/"
    masks_dir = "./data/dl/cropped/train_val_masks/"

    # Initial Preprocess
    raw_location = "./data/dl/raw/train/"
    raw_masks_dir = "./data/dl/raw/masks/"

    raw_data_dir = "test_images"
    raw_mask_dir = "test_masks"

    preprocessed_raw_location = "./data/dl/train_val_images/"
    preprocessed_masks_dir = "./data/dl/cropped/train_val_masks/"

    test_image_location = "./data/dl/cropped/test_images/*.png"
    test_mask_location = "./data/dl/cropped/test_masks/*.png"

    train_dir_images = "train_images"
    train_dir_masks = "train_masks"
    val_dir_images = "val_images"
    val_dir_masks = "val_masks"

    models_dir = "./data/models/final"
