from ultralytics import YOLO
import sys
from pathlib import Path
import pandas as pd

TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]

def create_simple_solution():
    # TEST_IMAGES_PATH = 'dataset/images/01'
    # SAVE_PATH = '123/123.xml'd
    model = YOLO('best.pt')
    all_res = []
    for f in Path(TEST_IMAGES_PATH).glob('*.JPG'):
        image_id = f.name[:-len(f.suffix)]
        result_pred = model(f, conf=0.5, imgsz=512)
        for i in result_pred[0].boxes:

            result = {
                'image_id': 'id',
                'xc': 0.,
                'yc': 0.,
                'w': 0.,
                'h': 0.,
                'label': 0,
                'score': 0.
            }

            xc, yc, w, h = i.xywhn[0].tolist()
            score = round(i.conf.item(), 6)
            result['image_id'] = image_id
            result['xc'] = round(xc, 6)
            result['yc'] = round(yc, 6)
            result['w'] = round(w, 6)
            result['h'] = round(h, 6)
            result['label'] = int(i.cls.item())
            result['score'] = score
            all_res.append(result)

    test_df = pd.DataFrame(all_res, columns=['image_id', 'xc', 'yc', 'w', 'h', 'label', 'score'])
    test_df.to_csv(SAVE_PATH, index=False)



def main():
    create_simple_solution()

if __name__ == '__main__':
    main()

