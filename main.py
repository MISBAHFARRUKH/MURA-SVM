import pandas as pd
import numpy as np
import skimage.io
import random

class Data:
    def _init_(self, base_path):
        self.base_path = base_path

    def read_csv(self, file_name):
        return pd.read_csv(f"{self.base_path}{file_name}", header=None, names=['FilePath', 'Labels'])

    def load_paths_and_labels(self, paths_file, labels_file):
        paths_df = self.read_csv(paths_file)
        labels_df = self.read_csv(labels_file)
        # Merge paths and labels on FilePath assuming that 'FilePath' is common
        merged_df = pd.merge(paths_df, labels_df, on='FilePath', how='left')
        return merged_df

data = Data(base_path=r"C:\Users\Qanita\Desktop\Misbah\\")

# Load training and validation data
train_data = data.load_paths_and_labels('train_paths.csv', 'train_labels.csv')
valid_data = data.load_paths_and_labels('valid_paths.csv', 'valid_labels.csv')

def extract_pixels(data_df):
    pixels = []
    for index, row in data_df.iterrows():
        image_path = data.base_path + row['FilePath']
        try:
            image_gray = skimage.io.imread(image_path, as_gray=True)
            if row['Labels'] == 'positive':
                pixels.append(image_gray.flatten())
            elif row['Labels'] == 'negative':
                pixels.append(image_gray.flatten())
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
        except Exception as e:
            print(f"An error occurred for {image_path}: {str(e)}")
    return np.array(pixels)

# Extract pixels
train_pixels = extract_pixels(train_data)
valid_pixels = extract_pixels(valid_data)

# Save the pixel data
np.save(r"C:\Users\Qanita\Desktop\Misbah\train_pixels.npy", train_pixels)
np.save(r"C:\Users\Qanita\Desktop\Misbah\valid_pixels.npy", valid_pixels)
