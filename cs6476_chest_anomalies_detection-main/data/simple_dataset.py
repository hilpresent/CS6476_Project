import os
import pandas as pd
import shutil
from tqdm import tqdm


def make_df_with_unique_labels(df, labels_column_name):
    new_df = df.copy(deep=True)
    i = df.index[-1] + 1
    to_drop = []
    for ind in df.index:
        row_dict = df.loc[ind].to_dict()
        labels = row_dict[labels_column_name].split("|")
        if len(labels) > 1:
            to_drop.append(ind)
            for label in labels:
                row_dict[labels_column_name] = label
                row = pd.DataFrame(row_dict, index=[i])
                new_df = pd.concat([new_df, row])
                i += 1      
    new_df = new_df.drop(to_drop)
    return new_df


def make_dataset(
    dataset_path, # Path to final dataset
    images_path,  # Path to extracted images
    csv_file='Data_Entry_2017_v2020.csv',
    train_file='train_val_list.txt',
    test_file='test_list.txt'
    labels_column_name='Finding Labels',
    image_column_name='Image Index'):
    
    df = pd.read_csv(csv_file, delimiter=',', nrows=None)
    df = make_df_with_unique_labels(df, labels_column_name)

    # Train data
    with open(train_file, 'r') as f:
        train_images = set([line.strip() for line in f.readlines()])
    df_all_train = df[df[image_column_name].isin(train_images)]
    
    # Test data
    with open(test_file, 'r') as f:
        test_images = set([line.strip() for line in f.readlines()])
    df_test = df[df[image_column_name].isin(test_images)]

    # Approx match train/test distributions by adjusting 'No Finding' class
    all_train_label_counts = df_all_train[labels_column_name].value_counts()
    test_label_counts = df_test[labels_column_name].value_counts()
    test_ratio = test_label_counts.max() / test_label_counts.sum()
    max_samples = int(test_ratio*(all_train_label_counts.sum()-all_train_label_counts.max())/(1-test_ratio))
    df_train = df_all_train.copy(deep=True)
    to_drop = df_train[df_train[labels_column_name] == "No Finding"].index[max_samples:]
    df_train = df_train.drop(to_drop)

    for label in df[labels_column_name].unique():
        # Create Train/Val subdirectories for current class
        train_dir = os.path.join(dataset_path, 'train', label)
        val_dir = os.path.join(dataset_path, 'val', label)
        os.makedirs(train_dir, exist_ok=True)
        print(f'Created {train_dir}')
        os.makedirs(val_dir, exist_ok=True)
        print(f'Created {val_dir}')

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
