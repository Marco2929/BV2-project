import os
from ultralytics import YOLO

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(base_dir, 'data', 'train.yaml')
output_path = os.path.join(base_dir, 'results', 'detection')

model = YOLO()

if __name__ == '__main__':
    model.train(data=data_path,
                epochs=200,
                patience=50,
                project=output_path,
                device=0,
                fliplr=0.0,
                flipud=0.0,
                copy_paste=0.5,
                perspective=0.0001,
                batch=-1
                )
