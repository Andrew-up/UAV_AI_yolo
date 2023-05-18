# This is a sample Python script.
from matplotlib import patches

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from utils.DataLoader import DataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np

def viz_dataset(image, labels):
    image = image[:, :, ::-1]
    fig, ax = plt.subplots()
    dh, dw, channel = image.shape
    ax.imshow(image)
    for label in labels:
        ann_parts = label.split()
        class_idx = int(ann_parts[0])
        x_centre = float(ann_parts[1])
        y_centre = float(ann_parts[2])
        width = float(ann_parts[3])
        height = float(ann_parts[4])
        x_min = (x_centre - width / 2.0) * image.shape[1]  # перевод относительных координат в абсолютные
        y_min = (y_centre - height / 2.0) * image.shape[0]
        box_width = width * image.shape[1]
        box_height = height * image.shape[0]
        color = 'r'  # цвет обводки прямоугольника (на выбор)
        linewidth = 1  # жирность обводки прямоугольника (на выбор)
        edgecolor = color
        rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=linewidth, edgecolor=edgecolor,
                                 facecolor='none')
        ax.add_patch(rect)
    plt.show()

def print_hi(name):

    dataloader = DataGenerator(images_path='dataset/images/',
                               labels_path='dataset/labels/',
                               batch_size=4,
                               input_shape=(5472, 3648))

    img, label = dataloader.__getitem__(1)
    print(label[1])
    viz_dataset(img[1], label[1])
    # # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
