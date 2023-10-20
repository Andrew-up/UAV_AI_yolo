import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

from definitions import ROOT_DIR

# device = torch.device("cpu")


IMAGE = False
SAVE_VIDEO = False
print(os.path.dirname(os.path.abspath(__file__)))  # This is your Project Root

font = cv2.FONT_HERSHEY_SIMPLEX


def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])

    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))

    area_of_intersection = i_height * i_width

    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1

    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1

    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection

    iou = area_of_intersection / area_of_union

    return iou


def get_video():
    model = YOLO('best.pt')
    # model.to(device)

    prev_frame_time = 0

    if IMAGE:
        new_names = []
        for dirpath, dirnames, filenames in os.walk('dataset/test/images'):
            for filename in filenames:
                new_names.append(os.path.join(dirpath, filename))

        results = model(new_names, save=True)
        # print(new_names)

    if not IMAGE:
        input_video_path = 'dataset/test/video.mp4'
        cap = cv2.VideoCapture(input_video_path)

        # add
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        size = (frame_width, frame_height)

        if SAVE_VIDEO:
            result = cv2.VideoWriter('out.mp4',
                                     cv2.VideoWriter_fourcc(*'MP4V'),
                                     10, size)

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
                    cv2.putText(resultimg, f'NO DETECTION', (30, frame_height - 30), font, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    cv2.putText(resultimg, f'fps: {fps}', (30, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('img', resultimg)
                # add
                if SAVE_VIDEO:
                    result.write(resultimg)
                cv2.waitKey(1)
            else:
                break

        cap.release()
        # add
        if SAVE_VIDEO:
            result.release()
        cv2.destroyAllWindows()


def main():
    from ultralytics import YOLO
    model = YOLO('best.pt')
    metrics = model.val()


def calculate_iou_for_dataset(directory):
    from ultralytics import YOLO
    yolo_model = YOLO('best.pt')
    files_in_directory = os.listdir(directory)
    is_valid = False

    if 'images' in files_in_directory and 'labels' in files_in_directory:
        print('Directory is valid')
        is_valid = True
    else:
        print(f'No "images" or "labels" folder found in {directory}')

    all_iou_scores = []

    if is_valid:
        txt_files = sorted(
            [file.name for file in os.scandir(os.path.join(directory, 'labels')) if
             file.name.endswith('.txt') and file.is_file()],
            reverse=True)
        image_files = sorted(
            [file.name for file in os.scandir(os.path.join(directory, 'images')) if
             file.name.endswith('.jpg') and file.is_file()],
            reverse=True)

        if len(txt_files) == len(image_files):
            print('Number of txt_files matches the number of images_files')

        print(f'img files: {len(image_files)}')
        print(f'txt files: {len(txt_files)}')

        for index, txt_file_name in enumerate(txt_files):
            txt_file_path = os.path.join(directory, 'labels', txt_file_name)
            img_path = os.path.join(directory, 'images', image_files[index])

            with open(txt_file_path, mode='r') as file:
                predictions = yolo_model(img_path, iou=0.5)[0]

                if 'boxes' in dir(predictions):
                    result_list = predictions.boxes.xywhn.tolist()
                    result_list = [[round(num, 3) for num in sublist] for sublist in result_list]
                    for idx, item in enumerate(result_list):
                        item.append(int(predictions.boxes.cls.tolist()[idx]))
                        item.append(round(predictions.boxes.conf.tolist()[idx], 3))

                    true_labels = []

                    for idx, line in enumerate(file.read().splitlines()):
                        values = [float(val) for val in line.split()]
                        values.append(int(values[0]))
                        del values[0]
                        true_labels.append(values)

                    print(result_list)
                    print(true_labels)

                    result_iou_scores = []

                    for prediction in result_list:
                        iou_scores = []

                        for label in true_labels:
                            iou = get_iou(np.array(label, dtype=np.float32), np.array(prediction, dtype=np.float32))
                            iou_scores.append(round(iou, 3))

                        if iou_scores:
                            result_iou_scores.append(max(iou_scores))

                    difference = len(true_labels) - len(result_list)

                    if difference != 0:
                        if difference < 0:
                            difference *= -1
                        for _ in range(difference):
                            result_iou_scores.append(0)

                    # if len(result_list) + len(true_labels) == 0:
                    #     result_iou_scores.append(1)

                    print(f'{file.name} IOU mean: {np.mean(result_iou_scores)}')
                    if result_iou_scores:
                        all_iou_scores.append(np.mean(result_iou_scores))

                    print('-' * 30)

        print(f'Mean IOU of the model: {np.mean(all_iou_scores)}')
        yolo_model.val()


def draw_plot():
    df = pd.read_excel('table.xlsx')

    df = df.sort_values(by='mAP50')

    # Извлечение значений столбцов
    dataset = df['dataset']
    time = df['time'] / 100
    mIoU = df['mIoU']
    mAP50 = df['mAP50']
    mAP50_95 = df['mAP50-95']

    # Создание графика
    plt.figure(figsize=(10, 6))
    plt.plot(dataset, time, marker='o', label='Время')
    plt.plot(dataset, mIoU, marker='s', label='mIoU')
    plt.plot(dataset, mAP50, marker='^', label='mAP50')
    plt.plot(dataset, mAP50_95, marker='d', label='mAP50-95')

    # Добавление подписей осей и заголовка
    plt.xlabel('Датасеты использованные для обучения')
    plt.ylabel('Значения')
    plt.title('График точности распознавания')
    plt.tight_layout()

    # Добавление легенды
    plt.legend()

    # Показать график
    plt.show()


from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':

    draw_plot()
    # import numpy as np

    # print(np.mean([0.606, 0.105, 0.0]))
    # image = Image.new("RGB", (1920, 1080), (255, 255, 255))
    # draw = ImageDraw.Draw(image)
    # ground_truth_bbox = np.array([50, 50, 500, 500], dtype=np.float32)
    # prediction_bbox = np.array([100, 100, 600, 600], dtype=np.float32)
    # iou = get_iou(ground_truth_bbox, prediction_bbox)
    # rectangle1 = ground_truth_bbox.tolist()
    # rectangle2 = prediction_bbox.tolist()
    # font = ImageFont.truetype('arial.ttf', 20)
    # draw.text((rectangle1[0], rectangle1[1]-22), 'x11 y11', font=font, fill=(0, 0, 0))
    # draw.text((rectangle1[2]-60, rectangle1[3]), 'x12 y12', font=font, fill=(0, 0, 0))
    #
    # draw.text((rectangle2[0], rectangle2[1]-22), 'x21 y21', font=font, fill=(0, 0, 0))
    # draw.text((rectangle2[2]-60, rectangle2[3]), 'x22 y22', font=font, fill=(0, 0, 0))
    #
    #
    # draw.rectangle(rectangle1, outline="red", width=3)
    # draw.rectangle(rectangle2, outline="blue", width=3)
    # image.save("rectangles.png")
    # image.show()

    # print('IOU: ', iou)
    #

    # aaaaa = [[1.4444, 2.22222, 31.2222222, 4.00000444444], [13, 33, 3, 34]]
    # aaaaa = [[round(num, 3) for num in sublist] for sublist in aaaaa]
    # print(aaaaa)


    # p = os.path.join(ROOT_DIR, 'dataset', 'test')
    # calculate_iou_for_dataset(p)


    get_video()
#     main()
