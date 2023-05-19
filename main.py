from ultralytics import YOLO

import torch
def print_hi(name):
    print(torch.version.cuda)
    # Load a model
    # model = YOLO('yolov8n.yaml')  # build a new model from scratch
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Use the model
    results = model.train(data="data_yaml.yaml", epochs=10, imgsz=512, name='yolov8n_custom', batch=8 )  # train the model
    results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

    # pl
    # success = model.export(format='onnx')  # export the model to ONNX format
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
