import os
import yaml
import json
import glob
from tqdm import tqdm


def load_annotations(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def load_model(config, model_name):
    trained_weights_path = os.path.join(config["training"]["project"], model_name, "weights", "best.pt")
    if not os.path.exists(trained_weights_path):
        from huggingface_hub import snapshot_download

        repo_id = 'mavikulov/yolo-model'
        token = "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"
        local_folder = snapshot_download(
            repo_id=repo_id,
            token=token,
            local_dir=config["training"]["project"]
        )

        print(f'Files were downloaded in {local_folder}')

    weights_path = trained_weights_path

    print(f"Extracting weights from {weights_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model was not found in path: {weights_path}")
    return weights_path


def delete_pt_files(directory):
    pt_files = glob.glob(os.path.join(directory, "*.pt"))
    for file in pt_files:
        try:
            os.remove(file)
            print(f"Removed file: {file}")
        except Exception as e:
            print(f"Error during removing file {file}: {e}")


def convert_coco_to_yolo(json_path, images_dir, labels_dir):
    data = load_annotations(json_path)
    os.makedirs(labels_dir, exist_ok=True)
    images = {img["id"]: img for img in data["images"]}

    for ann in tqdm(data["annotations"]):
        image_id = ann["image_id"]
        image_info = images[image_id]
        image_width = image_info["width"]
        image_height = image_info["height"]

        x_min, y_min, width, height = ann["bbox"]

        x_center = (x_min + width / 2) / image_width
        y_center = (y_min + height / 2) / image_height
        width_norm = width / image_width
        height_norm = height / image_height

        class_id = ann["category_id"] - 1  
        yolo_line = f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n"

        image_name = image_info["file_name"].replace(".jpg", ".txt")
        label_path = os.path.join(labels_dir, image_name)
        with open(label_path, "a") as f:
            f.write(yolo_line)


def create_yaml_from_coco(json_path, output_yaml_path, data_path):
    annotations = load_annotations(json_path)
    categories = annotations.get("categories", [])
    nc = len(categories)
    names = [cat["name"] for cat in categories]
    
    yaml_data = {
        "path": data_path,  
        "train": "images/train",  
        "val": "images/valid",  
        "test": "images/test",  
        "nc": nc,  
        "names": names
    }
    
    with open(output_yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
