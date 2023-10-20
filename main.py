import os.path

import yaml
from ultralytics import YOLO

from definitions import ROOT_DIR



# ultralytics.checks()
# import torch
#
# print(torch.cuda.is_available())


def print_hi(name):
    config_yolo = 'config_yolo.yaml'
    file_model_name = 'config_model.yaml'
    default_folder = True
    default_folder_path = os.path.join(ROOT_DIR, 'default_config')
    if os.path.exists(os.path.join(ROOT_DIR, file_model_name)):
        with open(os.path.join(ROOT_DIR, file_model_name), mode='r') as file_yaml:
            listyaml = yaml.load(file_yaml, Loader=yaml.FullLoader)
            model_name = listyaml.get('model_name')
            default_folder = False
    else:
        default_folder = True
        print(f'Пользовательский файл конфигурации {os.path.join(ROOT_DIR, file_model_name)} не найден')
        with open(os.path.join(default_folder_path, file_model_name), mode='r') as file_yaml:
            listyaml = yaml.load(file_yaml, Loader=yaml.FullLoader)
            model_name = listyaml.get('model_name')

    if os.path.exists(os.path.join(ROOT_DIR, config_yolo)) and not default_folder:
        default_folder = False
    else:
        default_folder = True
        config_yolo = os.path.join(default_folder_path, config_yolo)
        print(f'Пользовательский файл конфигурации {os.path.join(ROOT_DIR, config_yolo)} не найден')

    if default_folder:
        print('=' * 30)
        print('НАСТРОЙКИ КОНФИГУРАЦИИ ИСПОЛЬЗУЕЮТСЯ ПО УМОЛЧАНИЮ')
        print(f'СКОПИРУЙТЕ ВСЕ ФАЙЛЫ ИЗ {default_folder_path} В РОДИТЕЛЬСКИЙ КАТАЛОГ')
        print('=' * 30)
    else:
        print('=' * 30)
        print('ИСПОЛЬЗУЮТСЯ НАСТРОЙКИ КОНФИГУРАЦИИ ПОЛЬЗОВАТЕЛЯ')
        print('=' * 30)

    print(model_name)
    print(config_yolo)
    model = YOLO(model_name)  # build a new model from scratch
    train_process = model.train(cfg=config_yolo, data="dataset/data.yaml")  # train the model


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
