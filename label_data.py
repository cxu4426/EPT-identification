"""
Reads all images and puts image information into bugs.csv
    Headers: filename,semantic_label,numeric_label,partition
Also partitions the files into train/val/test, ensuring one of each in each class
"""
import os
import random
import pandas as pd
from utils import config

def create_csv(folder_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    image_path = os.path.join(folder_path, config("image_path"))

    # List all image files in the folder
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Check if the CSV file already exists
    csv_file_path = os.path.join(folder_path, config("csv_file"))
    existing_filenames = set()

    if os.path.exists(csv_file_path):
        # If the CSV file exists, read existing filenames
        df_existing = pd.read_csv(csv_file_path)
        existing_filenames = set(df_existing['filename'])

    # List of semantic labels so the user only has to input the numeric label
    labels = ['Ephemeroptera', 'Plecoptera', 'Trichoptera']

    # Filter out images already in the CSV
    new_images = [img for img in image_files if img not in existing_filenames]
    if not new_images:
        print("No new images. CSV will not update.")
        return

    # Shuffle the new images
    random.shuffle(new_images)

    # Create an empty dictionary to store partition data
    partition_data = {'train': [], 'validation': [], 'test': []}

    # Iterate over each class and distribute samples to each partition
    for label_num in ['0', '1', '2']:
        # Filter images for the current class
        class_images = [img for img in new_images if img.startswith(labels[int(label_num)][0].lower())]

        # Calculate sizes for train/val/test for the current class
        class_total_size = len(class_images)
        class_val_size = max(1, int(val_ratio * class_total_size))
        class_test_size = max(1, int(test_ratio * class_total_size))
        class_train_size = class_total_size - class_val_size - class_test_size

        # Partition the data for the current class
        partition_data['train'].extend(class_images[:class_train_size])
        partition_data['validation'].extend(class_images[class_train_size:class_train_size + class_val_size])
        partition_data['test'].extend(class_images[-class_test_size:])

    data = []
    for partition, partition_files in partition_data.items():
        for image_file in partition_files:
            letter = image_file[0].lower()
            if letter == 'e':
                numeric_label = '0'
            elif letter == 'p':
                numeric_label = '1'
            elif letter == 't':
                numeric_label = '2'
            else:
                print(f"Invalid filename: {image_file}. Skipping.")
                continue

            semantic_label = labels[int(numeric_label)]
            
            data.append({
                'filename': image_file,
                'semantic_label': semantic_label,
                'numeric_label': numeric_label,
                'partition': partition
            })
            existing_filenames.add(image_file)

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Sort DataFrame by 'filename'
    df = df.sort_values(by='filename')

    # Write DataFrame to CSV
    df.to_csv(csv_file_path, index=False)

    print(f"CSV file updated at: {csv_file_path}")

def main():
    # Specify the folder path
    folder_path = os.getcwd()
    # Call the function to create the CSV
    create_csv(folder_path)

if __name__ == "__main__":
    main()
