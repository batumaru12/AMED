import argparse
import os
from pathlib import Path
import torch
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

# DETR関連のコードをインポート
from models.detr import build
from models_mae import mae_vit_base_patch16  # MAEモデルをロード

def load_model_and_postprocessors(args):
    """
    DETRモデルと後処理(PostProcess)をロードする。

    Parameters:
        args (argparse.Namespace): 引数パーサ。

    Returns:
        model (torch.nn.Module): DETRモデル。
        postprocessors (dict): 後処理用の辞書。
    """
    # MAEバックボーンを使ったモデルを構築
    if args.backbone == "mae":
        mae_model = mae_vit_base_patch16()
        checkpoint = torch.load(args.mae_weights_path, map_location="cpu")
        mae_model.load_state_dict(checkpoint["model"], strict=False)
        mae_model.eval()
        args.mae_model = mae_model  # モデルをargsに保存してbuild関数に渡す

    model, _, postprocessors = build(args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()  # 推論モードに設定
    return model, postprocessors

def detect_and_visualize(model, postprocessors, image_folder, output_folder):
    """
    フォルダ内の画像に対してDETRを用いて物体検出を行い、結果を可視化して保存する。

    Parameters:
        model (torch.nn.Module): DETRモデル。
        postprocessors (dict): 後処理用の辞書。
        image_folder (str): 入力画像が格納されたフォルダ。
        output_folder (str): 検出結果を保存するフォルダ。
    """
    # 入力フォルダ内の画像ファイルを取得
    image_paths = list(Path(image_folder).glob('*.png'))
    
    # 出力フォルダを作成
    os.makedirs(output_folder, exist_ok=True)

    # tqdmで進捗表示
    for image_path in tqdm(image_paths, desc="Processing images"):
        # 画像をロードして前処理
        image = Image.open(image_path).convert('RGB')
        img_tensor = F.to_tensor(image).unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

        # モデルで推論
        with torch.no_grad():
            outputs = model(img_tensor)
            target_sizes = torch.tensor([[image.height, image.width]])
            results = postprocessors['bbox'](outputs, target_sizes)

        # 推論結果を取得
        boxes = results[0]['boxes'].cpu().numpy()  # バウンディングボックス
        scores = results[0]['scores'].cpu().numpy()  # スコア
        labels = results[0]['labels'].cpu().numpy()  # ラベル

        # 高スコアの結果だけを選択 (例: スコアが0.7以上)
        threshold = 0.3
        keep = scores > threshold
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # 検出結果を画像に重ねて描画
        draw = ImageDraw.Draw(image)
        for box, label, score in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min), f"Class {label}, Score: {score:.2f}", fill="red")

        # 結果画像を保存
        output_path = Path(output_folder) / image_path.name
        image.save(output_path)

if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description="DETRによる物体検出結果の可視化")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_dir", type=str, default="image_result", help="Folder to save detection results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes including background")
    parser.add_argument("--backbone", type=str, default="mae", help="Backbone type (e.g., mae, resnet50)")
    parser.add_argument("--mae_weights_path", type=str, required=True, help="Path to the MAE weights checkpoint")

    # モデル構築に必要な引数を追加
    parser.add_argument("--dataset_file", type=str, default="coco", help="Dataset type (e.g., coco)")
    parser.add_argument("--masks", action="store_true", help="Enable training with segmentation masks")
    parser.add_argument("--aux_loss", action="store_true", help="Enable auxiliary decoding losses (default: False)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
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

    args = parser.parse_args()

    # モデルと後処理をロード
    print("Loading model and postprocessors...")
    model, postprocessors = load_model_and_postprocessors(args)

    # 画像の検出と可視化を実行
    print("Running detection and visualization...")
    detect_and_visualize(model, postprocessors, args.image_dir, args.output_dir)

    print("Processing completed!")
