import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2
from tqdm import tqdm
import tensorflow as tf

'''
TODO:
Convert labels into list of it's indices
'''


def compileData(data_path, make_new=False):
    BUILD_DATA = make_new
    # Size of resized images (RESIZE_TO x RESIZE_TO)
    RESIZE_TO = 30
    if BUILD_DATA:
        list = os.listdir(data_path)
        dataset = []
        plot_dataset = []
        print("Concatenating images into matrix...")

        # Cycles through label folders (0,1,2,3) in the data folder
        for label_data in list:
            _, _, files = next(os.walk(os.path.join(data_path, label_data)))
            print("{} Images of label: {}".format(len(files),label_data))

            # Cycles though images in each label folder
            for im in tqdm(files):
                img = mpimg.imread(os.path.join(data_path, label_data, im))
                # Resizes images
                img = cv2.resize(img, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_AREA)
                dataset.append([img, int(label_data)])
                plot_dataset.append([img, int(label_data)])
        # Normalizes the images in the train_images dataset
        dataset = [[normalize(data[0]), int(data[1])] for data in dataset]
        # Shuffle dataset
        np.random.shuffle(dataset)
        np.random.shuffle(plot_dataset)
        print('Images in dataset: {}'.format(len(dataset)))
        # Save dataset
        np.save("training_data.npy", dataset)
    else:
        # If not building dataset, then just load it from .npy file
        dataset = np.load('training_data.npy', allow_pickle=True)
        plot_dataset = []
    return dataset, plot_dataset


def normalize(image):
    return image/255.0


def showTestImage(train_images):
    plt.figure()
    plt.title("Label: {}".format(train_images[0][1]))
    plt.imshow(train_images[0][0])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def example_plot(train_images):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i][0], cmap=plt.cm.binary)
        plt.xlabel(str(train_images[i][1]))
    plt.show()


if __name__ == "__main__":
    datapath = 'data'
    _, plot_dataset = compileData(data_path=datapath, make_new=True)
    showTestImage(plot_dataset)
    example_plot(plot_dataset)
