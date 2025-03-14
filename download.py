import os
import shutil


if __name__ == "__main__":
    from roboflow import Roboflow
    api_key = input("Enter your api_key: ")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("michael-gorrg").project("stalcraft-flcil")
    version = project.version(9)
    dataset = version.download("coco")
    data_path = "data"
    os.rename("stalcraft-9", data_path)

    with open(f'{data_path}/dataset.yaml', 'w') as file:
        pass

    for split in ['valid', 'test', 'train']:
        shutil.move(f'{data_path}/{split}', f'{data_path}/images/{split}')
        os.makedirs(os.path.join(data_path, 'labels', split), exist_ok=True)
    