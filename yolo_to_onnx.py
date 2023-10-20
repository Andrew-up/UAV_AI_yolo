from ultralytics import YOLO

providers = [
    ('TensorrtExecutionProvider', {
        'trt_engine_cache_enable': True,
    }),
    'CUDAExecutionProvider'
]


def main():
    model = YOLO('best.pt')
    model.export(imgsz=640, format='onnx', opset=15, simplify=True)


if __name__ == '__main__':
    main()
