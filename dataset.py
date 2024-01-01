"""
Bugs Dataset
    Class wrapper for interfacing with the dataset of bug images
    Usage: python dataset.py
"""
import os
import pandas as pd
import numpy as np
import random
from shutil import copyfile
from PIL import Image
from utils import config

def read_csv(csv_file):
    return pd.read_csv(csv_file)

def resize(img_path):
    """Resize and replace single image"""
    img = Image.open(img_path)
    new_img = img.resize((config("image_dim"), config("image_dim")))
    new_img.save(img_path)

def augment_data(input_folder, output_folder, n, imgs, keep_img=False):
    """Augment training images

    :input_folder: path to original images
    :output_folder: path to store augmented images
    :n: number of augmented images from each original image
    :imgs: list of filenames

    Augmentations include grayscale, rotation, and Gaussian noise
    """
    os.makedirs(output_folder, exist_ok=True)

    for img in imgs:
        img_path = os.path.join(input_folder, img)

        image = Image.open(img_path)

        aug_img_path = os.path.join(output_folder, img)

        if keep_img:
            # Copy the original image to the output folder
            image.save(aug_img_path)

        # Apply augmentations
        for i in range(n):
            # Grayscale
            grayscale_image = image.convert("L")
            grayscale_output_path = os.path.join(output_folder, f"{img.split('.')[0]}_grayscale_{i}.{img.split('.')[-1]}")
            grayscale_image.save(grayscale_output_path)

            # Rotation
            rotation_angle = random.uniform(-30, 30)
            rotated_image = image.rotate(rotation_angle)
            rotated_output_path = os.path.join(output_folder, f"{img.split('.')[0]}_rotated_{i}.{img.split('.')[-1]}")
            rotated_image.save(rotated_output_path)

            # Gaussian Noise
            noisy_image = add_gaussian_noise(image)
            noisy_output_path = os.path.join(output_folder, f"{img.split('.')[0]}_noisy_{i}.{img.split('.')[-1]}")
            noisy_image.save(noisy_output_path)

def add_gaussian_noise(image):
    """Add Gaussian noise to an image"""
    mean = 0
    std = 25
    noisy_image = image.copy()
    width, height = noisy_image.size
    gaussian_noise = ImageEnhance.Brightness(Image.new('L', (width, height), int(mean + random.gauss(0, std)))).enhance(0.01)
    noisy_image.paste(Image.new('RGB', (width, height), (0, 0, 0)), (0, 0), gaussian_noise)
    return noisy_image

def normalize_images(image_paths):
    """
    Normalize a list of images.

    :param image_paths: List of file paths to images.
    :return: List of normalized image arrays.
    """
    images = []

    # Load images
    for path in image_paths:
        image = Image.open(path)
        image_array = np.array(image)
        images.append(image_array)

    # Calculate mean and standard deviation
    images_array = np.stack(images, axis=0)
    mean = np.mean(images_array, axis=(0, 1, 2))
    std = np.std(images_array, axis=(0, 1, 2))

    # Normalize images
    normalized_images = [(image_array - mean) / std for image_array in images]

    return normalized_images


def main():
    data_path = os.path.join(os.getcwd(), 'data')
    csv_path = os.path.join(data_path, 'bugs.csv')
    df = read_csv(csv_path)

    train_imgs = df.loc[df['partition'] == 'train', 'filename'].tolist()
    val_imgs = df.loc[df['partition'] == 'validation', 'filename'].tolist()
    test_imgs = df.loc[df['partition'] == 'test', 'filename'].tolist()

    img_path = os.path.join(data_path, 'images')
    aug_img_path = os.path.join(data_path, 'augmented_images')


    # # Specify paths and ratios
    # csv_file_path = 'path/to/your/csv/file.csv'
    # input_images_folder = 'path/to/your/images/folder'
    # output_augmented_folder = 'path/to/your/augmented/folder'

    # # Define augmentations (you can replace this with your own augmentations)
    # def rotate(img_path):
    #     # Add your rotation logic here
    #     pass

    # def grayscale(img_path):
    #     # Add your grayscale logic here
    #     pass

    # augmentations = [rotate, grayscale]

    # # Read CSV file (data is a dataframe)
    # data = read_csv(csv_file_path)

    # # TODO: get training images only and augment (use augment_data.py?)

if __name__ == "__main__":
    main()


# import os
# import random
# import numpy as np
# import pandas as pd
# import torch
# from matplotlib import pyplot as plt
# from imageio.v3 import imread
# from PIL import Image
# from torch.utils.data import Dataset, DataLoader
# from utils import config

# import rng_control


# def get_train_val_test_loaders(task, batch_size, **kwargs):
#     """Return DataLoaders for train, val and test splits.

#     Any keyword arguments are forwarded to the BugsDataset constructor.
#     """
#     tr, va, te, _ = get_train_val_test_datasets(task, **kwargs)

#     tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
#     va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
#     te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

#     return tr_loader, va_loader, te_loader, tr.get_semantic_label


# def get_challenge(task, batch_size, **kwargs):
#     """Return DataLoader for challenge dataset.

#     Any keyword arguments are forwarded to the BugsDataset constructor.
#     """
#     tr = BugsDataset("train", task, **kwargs)
#     ch = BugsDataset("challenge", task, **kwargs)

#     standardizer = ImageStandardizer()
#     standardizer.fit(tr.X)
#     tr.X = standardizer.transform(tr.X)
#     ch.X = standardizer.transform(ch.X)

#     tr.X = tr.X.transpose(0, 3, 1, 2)
#     ch.X = ch.X.transpose(0, 3, 1, 2)

#     ch_loader = DataLoader(ch, batch_size=batch_size, shuffle=False)
#     return ch_loader, tr.get_semantic_label


# def get_train_val_test_datasets(task="default", **kwargs):
#     """Return BugsDatasets and image standardizer.

#     Image standardizer should be fit to train data and applied to all splits.
#     """
#     tr = BugsDataset("train", task, **kwargs)
#     va = BugsDataset("val", task, **kwargs)
#     te = BugsDataset("test", task, **kwargs)

#     # Resize
#     # You may want to experiment with resizing images to be smaller
#     # for the challenge portion. How might this affect your training?
#     # tr.X = resize(tr.X)
#     # va.X = resize(va.X)
#     # te.X = resize(te.X)

#     # Standardize
#     standardizer = ImageStandardizer()
#     standardizer.fit(tr.X)
#     tr.X = standardizer.transform(tr.X)
#     va.X = standardizer.transform(va.X)
#     te.X = standardizer.transform(te.X)

#     # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
#     tr.X = tr.X.transpose(0, 3, 1, 2)
#     va.X = va.X.transpose(0, 3, 1, 2)
#     te.X = te.X.transpose(0, 3, 1, 2)

#     return tr, va, te, standardizer


# def resize(X):
#     """Resize the data partition X to the size specified in the config file.

#     Use bicubic interpolation for resizing.

#     Returns:
#         the resized images as a numpy array.
#     """
#     image_dim = config("image_dim")
#     image_size = (image_dim, image_dim)
#     resized = []
#     for i in range(X.shape[0]):
#         xi = Image.fromarray(X[i]).resize(image_size, resample=2)
#         resized.append(xi)
#     resized = [np.asarray(im) for im in resized]
#     resized = np.array(resized)

#     return resized


# class ImageStandardizer(object):
#     """Standardize a batch of images to mean 0 and variance 1.

#     The standardization should be applied separately to each channel.
#     The mean and standard deviation parameters are computed in `fit(X)` and
#     applied using `transform(X)`.

#     X has shape (N, image_height, image_width, color_channel), where N is
#     the number of images in the set.
#     """

#     def __init__(self):
#         """Initialize mean and standard deviations to None."""
#         super().__init__()
#         self.image_mean = None
#         self.image_std = None

#     def fit(self, X):
#         """Calculate per-channel mean and standard deviation from dataset X.
#         Hint: you may find the axis parameter helpful"""
#         self.image_mean = np.mean(X, axis=(0,1,2))
#         self.image_std = np.std(X, axis=(0,1,2))

#     def transform(self, X):
#         """Return standardized dataset given dataset X."""
#         return (X - self.image_mean)/self.image_std


# class BugsDataset(Dataset):
#     """Dataset class for bug images."""

#     def __init__(self, partition, task="target", augment=False):
#         """Read in the necessary data from disk.

#         For parts 2, 3 and data augmentation, `task` should be "target".
#         For source task of part 4, `task` should be "source".

#         For data augmentation, `augment` should be True.
#         """
#         super().__init__()

#         if partition not in ["train", "val", "test", "challenge"]:
#             raise ValueError("Partition {} does not exist".format(partition))

#         np.random.seed(42)
#         torch.manual_seed(42)
#         random.seed(42)
#         self.partition = partition
#         self.task = task
#         self.augment = augment
#         # Load in all the data we need from disk
#         if task == "target" or task == "source":
#             self.metadata = pd.read_csv(config("csv_file"))
#         if self.augment:
#             print("Augmented")
#             self.metadata = pd.read_csv(config("augmented_csv_file"))
#         self.X, self.y = self._load_data()

#         self.semantic_labels = dict(
#             zip(
#                 self.metadata[self.metadata.task == self.task]["numeric_label"],
#                 self.metadata[self.metadata.task == self.task]["semantic_label"],
#             )
#         )

#     def __len__(self):
#         """Return size of dataset."""
#         return len(self.X)

#     def __getitem__(self, idx):
#         """Return (image, label) pair at index `idx` of dataset."""
#         return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

#     def _load_data(self):
#         """Load a single data partition from file."""
#         print("loading %s..." % self.partition)
#         df = self.metadata[
#             (self.metadata.task == self.task)
#             & (self.metadata.partition == self.partition)
#         ]
#         if self.augment:
#             path = config("augmented_image_path")
#         else:
#             path = config("image_path")

#         X, y = [], []
#         for i, row in df.iterrows():
#             label = row["numeric_label"]
#             image = imread(os.path.join(path, row["filename"]))
#             X.append(image)
#             y.append(row["numeric_label"])
#         return np.array(X), np.array(y)

#     def get_semantic_label(self, numeric_label):
#         """Return the string representation of the numeric class label.

#         (e.g., the numeric label 1 maps to the semantic label 'hofburg_imperial_palace').
#         """
#         return self.semantic_labels[numeric_label]


# if __name__ == "__main__":
#     np.set_printoptions(precision=3)
#     tr, va, te, standardizer = get_train_val_test_datasets(task="target", augment=False)
#     print("Train:\t", len(tr.X))
#     print("Val:\t", len(va.X))
#     print("Test:\t", len(te.X))
#     print("Mean:", standardizer.image_mean)
#     print("Std: ", standardizer.image_std)
