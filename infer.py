from ultralytics import YOLO
import argparse
import torch
import yaml
import cv2
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(config, model_name):
    pretrained_weights_path = os.path.join(os.getcwd(), "pretrained_models", f"{model_name}.pt")
    trained_weights_path = os.path.join(config["training"]["project"], model_name, "weights", "best.pt")

    if not os.path.exists(trained_weights_path):
        weights_path = pretrained_weights_path
        print(f"Used pretrained weights for {model_name}")
    else:
        weights_path = trained_weights_path
        print(f"Used pretrained weights for {model_name}")
    print(f"Extracting weights from {weights_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model was not found in path: {weights_path}")

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
    