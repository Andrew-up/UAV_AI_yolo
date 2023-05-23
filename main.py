from ultralytics import YOLO


def print_hi(name):
    # print(torch.version.cuda)

    # model = YOLO('yolov8s.yaml')  # build a new model from scratch
    # model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

    # results = model.train(data="data_yaml.yaml", epochs=5, imgsz=512, name='yolov8s_custom',
    #                       batch=32)  # train the model


    model = YOLO('runs/detect/yolov8s_custom/weights/best.pt')

    # model.export(format='onnx')


    # art = wandb.Artifact("my-object-detector", type="model")
    # art.add_file("runs/detect/yolov8s_custom/weights/best.pt")
    # wandb.log_artifact(art)

    results = model('test_data/images/1_001541.JPG', conf=0.1)

    print(results[0].boxes)
    print(results[0].boxes.conf)
    # print(results.probs)


    # pl
    # success = model.export(format='onnx')  # export the model to ONNX format
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
