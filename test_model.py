import time

import cv2
from ultralytics import YOLO


def get_video():
    model = YOLO('runs/detect/yolov8n_custom9/weights/best.pt')

    # image = cv2.imread('dataset/test/images/Image_013376_jpg.rf.c6d1923eb359a5da7a83f4778262b8cc.jpg')

    # res = model(image)

    # cv2.imshow('img', res[0].plot())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    input_video_path = 'dataset/test/video2.mp4'
    prev_frame_time = 0
    cap = cv2.VideoCapture(input_video_path)
    font = cv2.FONT_HERSHEY_SIMPLEX

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    while (cap.isOpened()):
        ret, frame = cap.read()
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        if ret:
            x1, y1, x2, y2, conf, cls = 0, 0, 0, 0, 0, 0
            max_conf = 0
            # cv2.imshow("frame", frame)
            resultimg = frame.copy()
            results = model.predict(frame)
            names = model.names
            name_class = None
            for res in results[0].boxes:
                _, _, _, _, conf_temp, _ = res.data[0].tolist()
                if conf_temp > max_conf:
                    max_conf = conf_temp
                    x1, y1, x2, y2, conf, cls = res.data[0].tolist()
                    x1, y1, x2, y2, conf, cls = int(x1), int(y1), int(x2), int(y2), round(conf_temp, 3), int(cls)
                    name_class = names[int(res.cls)]

            if max_conf > 0 and name_class:
                ima = cv2.rectangle(resultimg, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(ima, f'conf: {conf} class: {name_class}', (x1, y1 + 30), font, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                cv2.putText(ima, f'fps: {fps}', (30, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                center = int((x1 + x2) / 2), int((y1 + y2) / 2)
                cv2.circle(ima, center, 30, (0, 0, 255), 2)
                resultimg = cv2.putText(ima, f'center: {center}', (30, frame_height - 30), font, 1, (0, 0, 255), 2,
                                        cv2.LINE_AA)
            else:
                cv2.putText(resultimg, f'NO DETECTION', (30, frame_height - 30), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(resultimg, f'fps: {fps}', (30, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow('img', resultimg)

            cv2.waitKey(1)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    model = YOLO('runs/detect/yolov8n_custom2/weights/best.pt')
    metrics = model.val()


if __name__ == '__main__':
    get_video()
    # main()
