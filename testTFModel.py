import  tensorflow as tf

def main():
    model = tf.keras.models.load_model('runs/detect/yolov8n_custom/weights/best_saved_model')
    print()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':


    main()