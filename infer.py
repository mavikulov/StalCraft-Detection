from ultralytics import YOLO
import argparse
import torch
import yaml
import cv2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(config):
    model = YOLO(config['inference']['model_path'])
    results = model.predict(
        source=config['inference']['image_path'],
        save=True,
        conf=config['inference']['conf'],
        device=device
    )
    
    for result in results:
        image_with_boxes = result.plot()
        cv2.imshow("Prediction", image_with_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/parameters_file.yaml', help='Path to YAML-file with configuration')
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    inference(config)
    