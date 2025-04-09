import argparse
import os
import json
from pathlib import Path
import torch
from torchvision.transforms import functional as F
from PIL import Image
from tqdm import tqdm
from models.detr import build  # DETRの公式実装に基づく
import numpy as np

def load_model_and_postprocessors(args):
    model, _, postprocessors = build(args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 推論モードに設定
    return model, postprocessors

def generate_pseudo_labels(model, postprocessors, image_folder, output_json):
    image_paths = list(Path(image_folder).glob('*.png'))
    pseudo_labels = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": f"class_{i}"} for i in range(1, args.num_classes + 1)]
    }
    
    annotation_id = 0  # アノテーションIDのカウンタ

    for image_id, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        img_tensor = F.to_tensor(image).unsqueeze(0)

        # モデルで推論
        with torch.no_grad():
            outputs = model(img_tensor)
            target_sizes = torch.tensor([[height, width]])
            results = postprocessors['bbox'](outputs, target_sizes)

        # 推論結果の取得
        boxes = results[0]['boxes'].cpu().numpy()  # バウンディングボックス
        scores = results[0]['scores'].cpu().numpy()  # スコア
        labels = results[0]['labels'].cpu().numpy()  # ラベル

        # 高スコアの結果だけを選択 (例: スコアが0.7以上)
        threshold = 0.7
        keep = scores > threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # COCOフォーマットの画像情報
        pseudo_labels["images"].append({
            "id": int(image_id),  # JSON用にintに変換
            "file_name": image_path.name,
            "width": int(width),
            "height": int(height)
        })

        # COCOフォーマットのアノテーション作成
        for box, label, score in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = box
            width, height = x_max - x_min, y_max - y_min
            pseudo_labels["annotations"].append({
                "id": int(annotation_id),  # intに変換
                "image_id": int(image_id),
                "category_id": int(label),  # intに変換
                "bbox": [float(x_min), float(y_min), float(width), float(height)],  # floatに変換
                "score": float(score),  # floatに変換
                "area": float(width * height),  # floatに変換
                "iscrowd": 0
            })
            annotation_id += 1

    # JSONファイルに保存
    with open(output_json, "w") as f:
        json.dump(pseudo_labels, f, indent=4)

    print(f"Pseudo labels saved to {output_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pseudo labels with DETR")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_json", type=str, default="pseudo_labels.json", help="Path to output JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes including background")

    parser.add_argument("--dataset_file", type=str, default="coco", help="Dataset type (e.g., coco)")
    parser.add_argument("--masks", action="store_true", help="Enable training with segmentation masks")
    parser.add_argument("--aux_loss", action="store_true", help="Enable auxiliary decoding losses (default: False)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone model name (e.g., resnet50)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension of transformer hidden layers")
    parser.add_argument("--num_queries", type=int, default=10, help="Number of object queries")
    parser.add_argument("--position_embedding", type=str, default="sine", choices=("sine", "learned"), help="Type of positional embedding to use")
    parser.add_argument("--dilation", action="store_true", help="If true, use dilation in the last convolutional block")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for the transformer")
    parser.add_argument("--nheads", type=int, default=8, help="Number of attention heads in the transformer")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of the feedforward network in the transformer")
    parser.add_argument("--enc_layers", type=int, default=6, help="Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", type=int, default=6, help="Number of decoding layers in the transformer")
    parser.add_argument("--pre_norm", action="store_true", help="If true, apply layer normalization before attention and FFN")
    parser.add_argument("--set_cost_class", type=float, default=1.0, help="Class prediction cost for Hungarian matcher")
    parser.add_argument("--set_cost_bbox", type=float, default=5.0, help="BBox L1 prediction cost for Hungarian matcher")
    parser.add_argument("--set_cost_giou", type=float, default=2.0, help="GIoU cost for Hungarian matcher")
    parser.add_argument("--bbox_loss_coef", type=float, default=5.0, help="Weight for bounding box loss")
    parser.add_argument("--giou_loss_coef", type=float, default=2.0, help="Weight for GIoU loss")
    parser.add_argument("--eos_coef", type=float, default=0.1, help="Weight for no-object class")
    parser.add_argument('--mae_weights_path', default=None, type=str)
    parser.add_argument("--mae_mask_ratio", default=0.75, type=float)
    
    args = parser.parse_args()

    # モデルと後処理をロード
    print("Loading model and postprocessors...")
    model, postprocessors = load_model_and_postprocessors(args)

    # 疑似ラベルの生成
    print("Generating pseudo labels...")
    generate_pseudo_labels(model, postprocessors, args.image_dir, args.output_json)

    print("Processing completed!")
