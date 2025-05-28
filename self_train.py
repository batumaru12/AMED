import subprocess
import argparse
import os

num_loop = 10 #ループ回数

for i in range(num_loop):
    print(f"\n======================loop {i+1} start======================")

    python_path = "" #python環境実行ファイル
    train_output_dir = f"./train_weights/self_train_loop{i+1}/"
    epochs = 800
    if i == 0:
        coco_path = "./coco"
    else:
        coco_path = f"./save_json_folder/self_train_loop{i}"
    subprocess.run([
        python_path,
        "train.py",
        "--batch_size", "64",
        "--epochs", str(epochs),
        "--lr_drop", "640",
        "--num_classes", "2",
        "--backbone", "usemae",
        "--num_queries", "10",
        "--annotations_path", coco_path,
        "--device", "cuda",
        "--output_dir", train_output_dir,
        "--mae_weights_path", "./mae_weights/mask_75.pth"
    ], check=True)

    for i in range(epochs-1):
        os.remove(f"./{train_output_dir}/checkpoint_epoch_{epochs+1:03}.pth")
    
    model_path = f"{train_output_dir}/best_model.pth"
    json_path = f"./save_json_folder/self_train_loop{i+1}/instances_train2017.json"
    image_output_dir = f"./image_output/self_train_loop{i+1}/train/"
    subprocess.run([
        python_path,
        "detection_result.py",
        "--image_dir", "./coco/train2017/",
        "--output_dir", image_output_dir,
        "--model_path", model_path,
        "--backbone", "usemae",
        "--device", "cuda",
        "--mae_weights_path", "./mae_weights/mask_75.pth"
        "--save_json",
        "--json_path", json_path
    ], check=True)

    json_path = f"./save_json_folder/self_train_loop{i+1}/instances_val2017.json"
    image_output_dir = f"./image_output/self_train_loop{i+1}/val/"
    subprocess.run([
        python_path,
        "detection_result.py",
        "--image_dir", "./coco/val2017/",
        "--output_dir", image_output_dir,
        "--model_path", model_path,
        "--backbone", "usemae",
        "--device", "cuda",
        "--mae_weights_path", "./mae_weights/mask_75.pth"
        "--save_json",
        "--json_path", json_path
    ], check=True)

    json_path = f"./save_json_folder/self_train_loop{i+1}/instances_test2017.json"
    image_output_dir = f"./image_output/self_train_loop{i+1}/test/"
    subprocess.run([
        python_path,
        "detection_result.py",
        "--image_dir", "./coco/test2017/",
        "--output_dir", image_output_dir,
        "--model_path", model_path,
        "--backbone", "usemae",
        "--device", "cuda",
        "--mae_weights_path", "./mae_weights/mask_75.pth"
        "--save_json",
        "--json_path", json_path
    ], check=True)

    print(f"\n=======================loop {i+1} end======================")
