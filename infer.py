from ultralytics import YOLO
from utils import load_model
import argparse
import torch
import yaml
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(config, model_name):
    weights_path = load_model(config, model_name)
    model = YOLO(weights_path)
    results = model.predict(
        source=config['inference']['image_paths'],
        save=True,
        conf=config['inference']['conf'],
        device=device,
        project='my_results',
        name='inference'
    )
    
    for result in results:
        image_with_boxes = result.plot()
        cv2.imshow("Prediction", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str, 
        default='configs/parameters_file.yaml',
        help='Path to YAML-file with configuration'
    )

    parser.add_argument(
        "--model",
        type=str, 
        required=True, 
        choices=["yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="Model name (choose from: yolov8s, yolov8m, yolov8l, yolov8x)"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    inference(config, args.model)
    