import os
import pandas as pd
import shutil
from tqdm import tqdm

def make_dataset(
    dataset_path, # Path to final dataset
    images_path,  # Path to extracted images
    train_val_split=0.8,
    csv_file='Data_Entry_2017_v2020.csv',
    labels_column_name='Finding Labels',
    image_column_name='Image Index'):
    
    df = pd.read_csv(csv_file, delimiter=',', nrows=None)
    labels = [label for label in df[label_column_name].unique() if "|" not in label]
    
    for label in labels:
        # Create Train/Val subdirectories for current class
        train_dir = os.path.join(dataset_path, 'train', label)
        val_dir = os.path.join(dataset_path, 'val', label)
        os.makedirs(train_dir, exist_ok=True)
        print(f'Created {train_dir}')
        os.makedirs(val_dir, exist_ok=True)
        print(f'Created {val_dir}')

        # Select entries
        tmp = df[df[label_column_name] == label] # Add shuffle
        split = int(tmp.shape[0]*train_val_split)

        # Train
        for img in tqdm(tmp[image_column_name][:split]):
            src_img = os.path.join(images_path, img)
            dst_img = os.path.join(train_dir, img)
            shutil.copy(src_img, dst_img)

        # Val
        for img in tqdm(tmp[image_column_name][split:]):
            src_img = os.path.join(images_path, img)
            dst_img = os.path.join(val_dir, img)
            shutil.copy(src_img, dst_img)
