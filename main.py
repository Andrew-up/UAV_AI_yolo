from ultralytics import YOLO


def print_hi(name):
    # print(torch.version.cuda)
    # # Load a model
    model = YOLO('yolov8s.yaml')  # build a new model from scratch
    model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)
    #
    #
    # # model.export(format='pb')
    # # Use the model
    # results = model.train(data="data_yaml.yaml", epochs=5, imgsz=512, name='yolov8s_custom',
    #                       batch=32)  # train the model
    # wandb.finish()
    # results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
    # model = YOLO('runs/detect/yolov8s_custom3/weights/best.pt')

    model.export(format='onnx')


    # art = wandb.Artifact("my-object-detector", type="model")
    # art.add_file("runs/detect/yolov8s_custom/weights/best.pt")
    # wandb.log_artifact(art)

    # model.predict('testImg', save=True, imgsz=512, conf=0.3)

    # pl
    # success = model.export(format='onnx')  # export the model to ONNX format
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
