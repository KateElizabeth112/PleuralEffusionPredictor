# a script to preprocess the CheXpert dataset, mainly resizing of images to equal dimensions so that thet are suitable for training a ResNet classifier on them. 
import os
import numpy as np
import pandas as pd
from skimage.transform import resize
import matplotlib.pyplot as plt

def preprocessCheXpert(data_dir, image_size, split="train", plot=False):
    # check that the splt is either "train", "valid" or "test"
    if split not in ["train", "valid"]:
        raise ValueError('split must be either "train", "valid"')

    # read the csv file with the labels
    # check that the csv file exists
    csv_path = os.path.join(data_dir, f"{split}_reduced.csv")
    if not os.path.exists(csv_path):
        raise ValueError(f"CSV file {csv_path} does not exist")
    
    df = pd.read_csv(csv_path)

    # print the number of rows in the dataframe
    print(f"Number of rows in the dataframe: {len(df)}")
    
    # create a directory to save the preprocessed images for this split
    split_output_dir = os.path.join(data_dir, f"{split}_resized_npy")
    if not os.path.exists(split_output_dir):
        os.makedirs(split_output_dir)

    # create a directory to save the jpeg versions of the preprocessed images for visualization purposes
    jpg_output_dir = os.path.join(data_dir, f"{split}_resized_jpg")
    if not os.path.exists(jpg_output_dir):
        os.makedirs(jpg_output_dir)
    
    # loop through the rows of the dataframe and preprocess each image
    for idx, row in df.iterrows():
        print(f"Processing image {idx} of {len(df)} for split {split}")
        img_id = row['image_id']
        img_path = os.path.join(data_dir, f"{split}_npy/img_{format(idx, '05d')}.npy")
        
        # load the image as a numpy array
        img = np.load(img_path)
        
        # resize the image using skimage's resize function, which preserves the pixel value distribution better than PIL's resize function. make sure to set anti_aliasing to True to avoid aliasing artifacts
        img_resized = resize(img, (image_size, image_size), anti_aliasing=True)

        # plot the resized image side by side with the original image to check that the resizing worked correctly. underneath each image show a histogram of the pixel values to check that the pixel value distribution is preserved after resizing
        if plot:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].imshow(img, cmap='gray')
            axs[0, 0].set_title('Original Image')
            axs[0, 1].imshow(img_resized, cmap='gray')
            axs[0, 1].set_title('Resized Image')
            axs[1, 0].hist(img.flatten(), bins=20, range=(0, 1))
            axs[1, 0].set_title('Original Image Pixel Value Distribution')
            axs[1, 1].hist(img_resized.flatten(), bins=20, range=(0, 1))
            axs[1, 1].set_title('Resized Image Pixel Value Distribution')
            plt.savefig(os.path.join(jpg_output_dir, f"img_{format(idx, '05d')}.jpg"))

        np.save(os.path.join(split_output_dir, f"img_{format(idx, '05d')}.npy"), img_resized)


def main():
    data_dir = '/Users/katephd/Documents/data/CheXpertSmall'
    image_size = 128

    preprocessCheXpert(data_dir, image_size, split="train")
    preprocessCheXpert(data_dir, image_size, split="valid")

if __name__ == "__main__":
    main()