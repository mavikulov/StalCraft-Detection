import gradio as gr
import cv2
import yaml
import argparse
import numpy as np
from ultralytics import YOLO
from utils import load_model


def detect_objects(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0]) 

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image 


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
        choices=["yolov8s", "yolov8m", "yolov8l"],
        help="Model name (choose from: yolov8s, yolov8m, yolov8l)"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    weights_path = load_model(config, args.model)
    model = YOLO(weights_path)

    demo = gr.Interface(
        fn=lambda image: detect_objects(image, model),
        inputs=gr.Image(type="numpy"),
        outputs=gr.Image(type="numpy"),
        title="StalCraft Object Detection",
        description="Загрузите или перетащите изображение, и YOLO предскажет объекты!"
    )

    demo.launch(share=True)
