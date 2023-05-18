import numpy as np
import cv2
import os
from keras.utils import Sequence
from PIL import Image


class DataGenerator(Sequence):
    def __init__(self, images_path, labels_path, batch_size, input_shape):
        self.images_path = images_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.image_list = []
        self.labels_list = []
        self.get_images_list()
        self.get_labels_list()


        print(len(self.image_list))
        print(len(self.labels_list))

    def get_images_list(self):
        for root, dirs, files in os.walk(self.images_path):
            for file in files:
                if file.lower().endswith('.jpg'):
                    self.image_list.append(os.path.normpath(os.path.join(root, file)))

    def get_labels_list(self):
        for root, dirs, files in os.walk(self.labels_path):
            for file in files:
                if file.lower().endswith('.txt'):
                    self.labels_list.append(os.path.normpath(os.path.join(root, file)))

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        labels = []

        for i in range(self.batch_size):
            # Получение изображение
            img = cv2.imread(os.path.join(batch_x[i]))
            img = cv2.resize(img, self.input_shape)
            images.append(img)
            print(os.path.join(batch_x[i]))
            # Получение аннотации

            with open(batch_y[i], 'r') as f:
                label = f.read().splitlines()
                labels.append(label)


        return images, labels
