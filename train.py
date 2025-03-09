import os
import time
import yaml
import torch
import shutil
import argparse
from ultralytics import YOLO
from utils import convert_coco_to_yolo, create_yaml_from_coco, delete_pt_files


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_yolo(config):
    for split in ['train', 'valid', 'test']:
        json_path = os.path.join(config['data']['images_dir'], split, '_annotations.coco.json')
        images_dir = os.path.join(config["data"]["images_dir"], split)
        labels_dir = os.path.join(config["data"]["labels_dir"], split)
        convert_coco_to_yolo(json_path, images_dir, labels_dir)
        
    create_yaml_from_coco(
        json_path=os.path.join(config["data"]["images_dir"], "train", "_annotations.coco.json"),
        output_yaml_path=config["data"]["yaml_path"],
        data_path=os.path.abspath(os.path.dirname(config["data"]["images_dir"]))
    )

    for model_name in config['pretrained_models']:
        name = model_name.split(".")[0]
        model_dir = os.path.join(config['training']['project'], name)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

        print(f"Training model: {model_name}")
        config['training']['name'] = name
        model = YOLO(model_name)

        results = model.train(
            data=config['data']['yaml_path'],
            epochs=config['training']['epochs'],
            batch=config['training']['batch'],
            imgsz=config['training']['imgsz'],
            device=device,
            project=config['training']['project'],
            name=name,
            verbose=True
        )
        
        metrics = model.val(
            data=config['data']['yaml_path'],
            split='test',
            conf=0.5,
            iou=0.5,
            device=device
        )
        
        print(f"Validationing Model: {model_name}")
        weights_path = os.path.join(config['training']['project'], name, "weights", "best.pt")
        print(f"Weights are saved in: {weights_path}")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/parameters_file.yaml", help="Path to YAML-file with configuration")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    train_yolo(config)

    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    delete_pt_files(script_dir)
