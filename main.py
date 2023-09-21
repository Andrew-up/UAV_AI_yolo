import ultralytics
from ultralytics import YOLO


# ultralytics.checks()
# import torch
#
# print(torch.cuda.is_available())


def print_hi(name):
    # pass
    model = YOLO('yolov8n.yaml')  # build a new model from scratch
    train_process = model.train(cfg='config_yolo.yaml', data="dataset/data.yaml")  # train the model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
