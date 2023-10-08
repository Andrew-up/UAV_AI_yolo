
from ultralytics import YOLO

def main():

    model = YOLO('best.pt')
    model.export(imgsz=640, format='onnx')


if __name__ == '__main__':
    main()
