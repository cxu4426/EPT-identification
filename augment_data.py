"""
Script to create an augmented dataset.
"""

import argparse
import csv
import glob
import os
import sys
import numpy as np
from scipy.ndimage import rotate
from imageio.v3 import imread, imwrite
from utils import config

import rng_control


def Rotate(deg=20):
    """Return function to rotate image."""

    def _rotate(img):
        """Rotate a random integer amount in the range (-deg, deg) (inclusive).

        Keep the dimensions the same and fill any missing pixels with black.

        :img: H x W x C numpy array
        :returns: H x W x C numpy array
        """
        # Randomly choose a rotation angle in the range (-deg, deg)
        angle = np.random.randint(-deg, deg)
        rotated_img = rotate(img, angle, reshape=False)

        return rotated_img

    return _rotate


def Grayscale():
    """Return function to grayscale image."""

    def _grayscale(img):
        """Return 3-channel grayscale of image.

        Compute grayscale values by taking average across the three channels.

        Round to the nearest integer.

        :img: H x W x C numpy array
        :returns: H x W x C numpy array

        """
        grayscale_img = np.mean(img, axis=-1, keepdims=True)
    
        # Round to the nearest integer
        grayscale_img = np.round(grayscale_img).astype(np.uint8)

        # Expand dimensions to make it H x W x 3
        grayscale_img = np.repeat(grayscale_img, 3, axis=-1)
        
        # avg_img = np.mean(img, axis=2, keepdims=True)
        # avg_img = np.rint(avg_img)

        # for i in range(len(avg_img)):
        #     for j in range(len(avg_img[i])):
        #         img[i][j][0] = avg_img[i][j]
        #         img[i][j][1] = avg_img[i][j]
        #         img[i][j][2] = avg_img[i][j]

        return grayscale_img

    return _grayscale

def Noise(scale=0.02):
    """Return function to add camera noise to image.

    :scale: Scale factor for noise intensity
    :returns: Function to add camera noise to image
    """

    def _noise(img):
        """Add Gaussian noise to the image.

        Randomly perturb pixel values with Gaussian noise in each channel.

        :img: H x W x C numpy array
        :returns: H x W x C numpy array
        """
        # Generate Gaussian noise with the same shape as the input image
        noise = np.random.normal(scale=scale, size=img.shape)

        # Add noise to each channel independently
        noisy_img = img + noise

        # Ensure pixel values are in the valid range [0, 255]
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        return noisy_img

    return _noise


def augment(filename, transforms, n=1, original=True):
    """Augment image at filename.

    :filename: name of image to be augmented
    :transforms: List of image transformations
    :n: number of augmented images to save
    :original: whether to include the original images in the augmented dataset or not
    :returns: a list of augmented images, where the first image is the original user chooses to include it

    """
    print(f"Augmenting {filename}")
    img = imread(filename)
    res = [img] if original else []
    for i in range(n):
        new = imread(filename)
        for transform in transforms:
            new = transform(new)
        res.append(new)
    return res


def main(args):
    """Create augmented dataset."""
    reader = csv.DictReader(open(args.input, "r"), delimiter=",")
    writer = csv.DictWriter(
        open(f"{args.datadir}/augmented_bugs.csv", "w"),
        fieldnames=["filename", "semantic_label", "numeric_label", "partition"],
    )
    augment_partitions = set(args.partitions)

    # TODO: change `augmentations` to specify which augmentations to apply
    # augmentations = [Rotate()]
    augmentations = [Noise()]

    writer.writeheader()
    os.makedirs(f"{args.datadir}/augmented/", exist_ok=True)
    for f in glob.glob(f"{args.datadir}/augmented/*"):
        print(f"Deleting {f}")
        os.remove(f)
    for row in reader:
        if row["partition"] not in augment_partitions:
            imwrite(
                f"{args.datadir}/augmented/{row['filename']}",
                imread(f"{args.datadir}/images/{row['filename']}"),
            )
            writer.writerow(row)
            continue
        imgs = augment(
            f"{args.datadir}/images/{row['filename']}",
            augmentations,
            n=1,  # TODO: choose how many times you want the augment the same image. Default is 1
            original=True,  # TODO: change to False to exclude original image.
        )
        for i, img in enumerate(imgs):
            fname = f"{row['filename'][:-4]}_aug_{i}.png"
            imwrite(f"{args.datadir}/augmented/{fname}", img)
            writer.writerow(
                {
                    "filename": fname,
                    "semantic_label": row["semantic_label"],
                    "partition": row["partition"],
                    "numeric_label": row["numeric_label"],
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to input CSV file", default="data/bugs.csv")
    parser.add_argument("--datadir", help="Data directory", default="data")
    parser.add_argument(
        "-p",
        "--partitions",
        nargs="+",
        help="Partitions (train|val|test)+ to apply augmentations to. Defaults to train",
        default=["train"],
    )
    main(parser.parse_args(sys.argv[1:]))
