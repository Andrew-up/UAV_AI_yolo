import numpy as np
import cv2
import os
from keras.utils import Sequence
from PIL import Image
import pandas as pd
import copy
class DataGenerator(Sequence):
    def __init__(self, images_path, batch_size, input_shape):
        self.images_path = images_path
        # self.labels_path = labels_path
        self.batch_size = batch_size
        self.input_shape = (input_shape[0], input_shape[1])
        self.image_list = []
        self.labels_list = []
        self.get_images_list()
        merge = []

        print(len(self.image_list))
        print(len(self.labels_list))

    def get_images_list(self):
        for root, dirs, files in os.walk(self.images_path):
            for file in files:
                if file.lower().endswith('.jpg'):
                    self.image_list.append(os.path.normpath(os.path.join(root, file)))
                if file.lower().endswith('.txt'):
                    self.labels_list.append(os.path.normpath(os.path.join(root, file)))

        self.image_list.sort()
        self.labels_list.sort()

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_list[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels_list[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        box_data = []

        for i in range(len(batch_x)):
            # Получение изображение
            img = cv2.imread(os.path.join(batch_x[i]))
            img = cv2.resize(img, self.input_shape)

            images.append(img)

            # Получение аннотации
            print(batch_y[i])
            data = np.loadtxt(batch_y[i])
            boxes = []
            # print(os.path.join(batch_x[i]))
            # print(len(data.shape))
            if len(data.shape) == 1:
                class_id, x, y, width, height = data
                x_min = int((x - width / 2) * self.input_shape[1])
                y_min = int((y - height / 2) * self.input_shape[0])
                x_max = int((x + width / 2) * self.input_shape[1])
                y_max = int((y + height / 2) * self.input_shape[0])
                boxes.append([x_min, y_min, x_max, y_max, int(class_id)])
            else:
                for j in range(len(data)):
                    class_id, x, y, width, height = data[j]
                    x_min = int((x - width / 2) * self.input_shape[1])
                    y_min = int((y - height / 2) * self.input_shape[0])
                    x_max = int((x + width / 2) * self.input_shape[1])
                    y_max = int((y + height / 2) * self.input_shape[0])
                    boxes.append([x_min, y_min, x_max, y_max, int(class_id)])

            box_data.append(np.array(boxes, dtype=np.float32))

        labels = copy.deepcopy(np.array(box_data))
        labels[..., 2:4] = labels[..., 2:4] - labels[..., 0:2]
        labels[..., 0:2] = labels[..., 0:2] + labels[..., 2:4] / 2

        image_data = np.array(images)
        box_data = np.array(box_data)
        # y_true = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)


        return images, labels
