import os.path

import yaml
from ultralytics import YOLO


# ultralytics.checks()
# import torch
#
# print(torch.cuda.is_available())


def print_hi(name):
    model_name = 'yolov8n.yaml'
    file_model_name = 'config_model.yaml'
    if os.path.exists(file_model_name):
        with open(file_model_name, mode='r') as file_yaml:

            listyaml = yaml.load(file_yaml, Loader=yaml.FullLoader)
            model_name = listyaml.get('model_name')
    else:
        print(f'FILE NOT FOUND: {file_model_name}')

    print(model_name)

    model = YOLO(model_name)  # build a new model from scratch
    train_process = model.train(cfg='config_yolo.yaml', data="dataset/data.yaml")  # train the model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
