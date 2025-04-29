import argparse
import os
from pathlib import Path
import torch
from torchvision import transforms as T
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

# DETR関連のコードをインポート
from models.detr import build  # これはDETRの公式実装に依存

def load_model_and_postprocessors(args):
    """
    DETRモデルと後処理(PostProcess)をロードする。
    """
    model, _, postprocessors = build(args)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(args.device)
    model.eval()
    return model, postprocessors

def get_transform():
    """
    DETR推論用画像変換（ImageNet正規化）。
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def detect_and_visualize(model, postprocessors, image_folder, output_folder, device, threshold):
    """
    DETRを用いて画像フォルダ内の物体検出と可視化。
    """
    image_paths = list(Path(image_folder).glob('*.png'))
    os.makedirs(output_folder, exist_ok=True)
    transform = get_transform()

    for image_path in tqdm(image_paths, desc="Processing images"):
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            target_sizes = torch.tensor([[image.height, image.width]], device=device)
            results = postprocessors['bbox'](outputs, target_sizes)

        boxes = results[0]['boxes'].cpu().numpy()
        scores = results[0]['scores'].cpu().numpy()
        labels = results[0]['labels'].cpu().numpy()

        keep = scores > threshold
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        draw = ImageDraw.Draw(image)
        for box, label, score in zip(boxes, labels, scores):
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
            draw.text((x_min, y_min), f"Class {label}, Score: {score:.2f}", fill="red")

        output_path = Path(output_folder) / image_path.name
        image.save(output_path)

def detect_and_visualize_video(model, postprocessors, video_dir, output_folder, device, threshold):
    os.makedirs(output_folder, exist_ok=True)
    transform = get_transform()

    video_paths = list(Path(video_dir).glob('*.avi'))

    for video_path in tqdm(video_paths, desc="Processing videos"):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Warning: Cannot open video {video_path}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        output_video_name = video_path.stem + '.avi'
        output_path = Path(output_folder) / output_video_name
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                target_sizes = torch.tensor([[image.height, image.width]], device=device)
                results = postprocessors['bbox'](outputs, target_sizes)

            boxes = results[0]['boxes'].cpu().numpy()
            scores = results[0]['scores'].cpu().numpy()
            labels = results[0]['labels'].cpu().numpy()

            keep = scores > threshold
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            for box, label, score in zip(boxes, labels, scores):
                x_min, y_min, x_max, y_max = box
                cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                text = f"Class {label}: {score:.2f}"
                cv2.putText(frame, text, (int(x_min), int(y_min) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            out.write(frame)

        cap.release()
        out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DETRによる物体検出結果の可視化")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("--output_dir", type=str, default="image_result", help="Folder to save detection results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of object classes (excluding background)")
    parser.add_argument("--threshold", type=float, default=0.5)

    # build() に必要なオプション
    parser.add_argument("--dataset_file", type=str, default="coco")
    parser.add_argument("--masks", action="store_true")
    parser.add_argument("--aux_loss", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_queries", type=int, default=10)
    parser.add_argument("--position_embedding", type=str, default="sine", choices=("sine", "learned"))
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--nheads", type=int, default=8)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--enc_layers", type=int, default=6)
    parser.add_argument("--dec_layers", type=int, default=6)
    parser.add_argument("--pre_norm", action="store_true")
    parser.add_argument("--set_cost_class", type=float, default=1.0)
    parser.add_argument("--set_cost_bbox", type=float, default=5.0)
    parser.add_argument("--set_cost_giou", type=float, default=2.0)
    parser.add_argument("--bbox_loss_coef", type=float, default=5.0)
    parser.add_argument("--giou_loss_coef", type=float, default=2.0)
    parser.add_argument("--eos_coef", type=float, default=0.1)
    parser.add_argument('--mae_weights_path', default=None, type=str)
    parser.add_argument("--mae_mask_ratio", default=0.75, type=float)

    parser.add_argument("--video", action="store_true")

    args = parser.parse_args()

    print("Loading model and postprocessors...")
    model, postprocessors = load_model_and_postprocessors(args)

    if args.video:
        print("Running detection and visualization...")
        detect_and_visualize_video(model, postprocessors, args.image_dir, args.output_dir, args.device, args.threshold)
    else:
        print("Running detection and visualization...")
        detect_and_visualize(model, postprocessors, args.image_dir, args.output_dir, args.device, args.threshold)

    print("Processing completed!")
