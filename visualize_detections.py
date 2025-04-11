import argparse
import os
from pathlib import Path
import torch
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T

from models import build_model
import util.misc as utils

def get_args():
    parser = argparse.ArgumentParser(description="DETR Inference and Visualization Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_dir", type=str, default="image_result", help="Folder to save detection results")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes including background")
    
    # モデル構築に必要なすべての引数を追加
    parser.add_argument("--dataset_file", type=str, default="coco", help="Dataset type (e.g., coco)")
    parser.add_argument("--masks", action="store_true", help="Enable training with segmentation masks")
    parser.add_argument("--aux_loss", action="store_true", help="Enable auxiliary decoding losses (default: False)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--backbone", type=str, default="resnet50", help="Backbone model name (e.g., resnet50)")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimension of transformer hidden layers")
    parser.add_argument("--num_queries", type=int, default=100, help="Number of object queries")
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
    
    # COCO形式でJSON保存オプション
    parser.add_argument("--save_json", action="store_true", help="If set, save results in COCO format JSON")
    return parser.parse_args()

def load_model(checkpoint_path, device, args):
    model, _, _ = build_model(args)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model

# シンプルな画像変換関数
def transform_image(image):
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  # バッチ次元を追加

def detect_image(model, image, device):
    img_tensor = transform_image(image).to(device)

    # 推論
    outputs = model(img_tensor)
    probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]

    # 信頼度の閾値を下げて0.5に設定
    keep = probas.max(-1).values > 0.3

    # デバッグ用に検出結果を表示
    print("Detection probabilities:", probas.max(-1).values)
    print("Keeping boxes:", keep.sum().item())

    boxes = outputs["pred_boxes"][0, keep].cpu()
    scores = probas[keep].max(-1).values.cpu()
    return boxes, scores

def visualize_and_save(image, boxes, output_path, scores=None):
    image_width, image_height = image.size  # PILの画像から幅・高さを取得
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for box, score in zip(boxes, scores):
        # バウンディングボックスをピクセル単位に変換
        x_c, y_c, w_c, h_c = box.detach().numpy()  # デタッチ＆NumPy変換
        # x = (x_c - 0.5 * w_c) * image_width  # 中心座標から左上角のxに変換
        # y = (y_c - 0.5 * h_c) * image_height  # 中心座標から左上角のyに変換
        # w = (x_c + 0.5 * w_c) * image_width
        # h = (y_c + 0.5 * h_c) * image_height

        x_min = (x_c - w / 2) * image_width  # 左上のx座標
        y_min = (y_c - h / 2) * image_height  # 左上のy座標
        w = w * image_width  # 幅をピクセル単位に変換
        h = h * image_height  # 高さをピクセル単位に変換

        print(image_width)
        print(image_height)
        print(x)
        print(y)
        print(w)
        print(h)

        # 描画
        # rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
        # ax.add_patch(rect)
        # ax.text(x, y, f"{score:.2f}", color="white", verticalalignment="top", bbox={"color": "red", "pad": 0})

        rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)
        ax.text(x_min, y_min, f"{score:.2f}", color="white", verticalalignment="top", bbox={"color": "red", "pad": 0})

    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)

def main(args):
    device = torch.device(args.device)
    model = load_model(args.model_checkpoint, device, args)

    output_dir = Path(args.output_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    # COCO形式の結果を保存するリスト
    results = []

    # 入力フォルダ内の全ての画像ファイルに対して推論を実行
    for img_path in Path(args.input_folder).glob("*.png"):  # 拡張子に合わせて変更可能
        image = Image.open(img_path)#.convert("RGB")
        boxes, scores = detect_image(model, image, device)

        # 結果を可視化して保存
        result_img_path = output_dir / f"{img_path.stem}_result.png"
        visualize_and_save(image, boxes, result_img_path, scores)

        # COCO形式の結果を構築
        for box, score in zip(boxes, scores):
            x, y, w, h = box.tolist()
            results.append({
                "image_id": img_path.stem,
                "category_id": 0,  # tumorを1として設定（0は背景）
                "bbox": [x, y, w, h],
                "score": float(score),
            })

    # COCO形式のJSONファイルを保存
    if args.save_json:
        json_path = output_dir / "detections.json"
        with open(json_path, "w") as f:
            json.dump(results, f)
        print(f"COCO format results saved to {json_path}")

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    args = get_args()
    main(args)
