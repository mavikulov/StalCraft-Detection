import os
import time
import yaml
import torch
import argparse
from ultralytics import YOLO
from utils import convert_coco_to_yolo, create_yaml_from_coco


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_yolo(config):
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(config['data']['images_dir'], split, '_annotations.coco.json')
        images_dir = os.path.join(config["data"]["images_dir"], split)
        labels_dir = os.path.join(config["data"]["labels_dir"], split)
        print(f"json_path = {json_path}")
        print(f"images_dir = {images_dir}")
        print(f"labels_dir = {labels_dir}")
        convert_coco_to_yolo(json_path, images_dir, labels_dir)
        
    create_yaml_from_coco(
        json_path=os.path.join(config["data"]["images_dir"], "train", "_annotations.coco.json"),
        output_yaml_path=config["data"]["yaml_path"],
        data_path=os.path.abspath(os.path.dirname(config["data"]["images_dir"]))
    )
    
    for model_name in config['pretrained_models']:
        print(f"Training model: {model_name}")
        model = YOLO(model_name)
        results = model.train(
            data=config['data']['yaml_path'],
            epochs=config['training']['epochs'],
            batch=config['training']['batch'],
            imgsz=config['training']['imgsz'],
            device=device,
            project=config['training']['project'],
            name=f"{model_name.split('.')[0]}_{time.strftime('%Y-%m-%d %H-%M-%S')}",
            verbose=True
        )
        
        metrics = model.val(
            data=config['data']['yaml_path'],
            split='test',
            conf=0.5,
            iou=0.5,
            device=device
        )
        
        print(f"Model: {model_name}")
        print(f"mAP@0.5: {metrics.box.map}")
        print(f"mAP@0.5:0.95: {metrics.box.map50_95}")
        print(f"Precision: {metrics.box.precision}")
        print(f"Recall: {metrics.box.recall}")
        print("-" * 50)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/parameters_file.yaml", help="Path to YAML-file with configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    print(os.path.abspath(os.path.dirname(config["data"]["images_dir"]))) 
    train_yolo(config)
        