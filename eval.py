import argparse
import torch
from pathlib import Path

from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import util.misc as utils
from engine import evaluate

def get_args():
    parser = argparse.ArgumentParser(description="DETR Evaluation Script")

    # データセットとモデル関連の引数
    parser.add_argument("--coco_path", type=str, default="./coco/", help="Path to COCO dataset directory")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory containing model checkpoints")
    parser.add_argument("--dataset_file", type=str, default="coco", help="Dataset type (e.g., coco)")
    parser.add_argument("--masks", action="store_true", help="Enable evaluation with segmentation masks")
    parser.add_argument("--aux_loss", action="store_true", help="Enable auxiliary decoding losses (default: False)")

    # 評価パラメータ
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for evaluation")
    parser.add_argument("--model_checkpoint", type=str, default="best_model.pth", help="Name of the model checkpoint to load for evaluation")

    # DETRモデルの設定
    parser.add_argument("--backbone", type=str, default="resnet50", help="Name of the backbone model (e.g., resnet50)")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes for COCO dataset")
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

    # マッチングコストと損失係数
    parser.add_argument("--set_cost_class", type=float, default=1.0, help="Class prediction cost for Hungarian matcher")
    parser.add_argument("--set_cost_bbox", type=float, default=5.0, help="BBox L1 prediction cost for Hungarian matcher")
    parser.add_argument("--set_cost_giou", type=float, default=2.0, help="GIoU cost for Hungarian matcher")
    parser.add_argument("--bbox_loss_coef", type=float, default=5.0, help="Weight for bounding box loss")
    parser.add_argument("--giou_loss_coef", type=float, default=2.0, help="Weight for GIoU loss")
    parser.add_argument("--eos_coef", type=float, default=0.1, help="Weight for no-object class")

    # 学習率関連の引数（評価には直接関係ないが、モデル構築に必要）
    parser.add_argument("--lr_backbone", type=float, default=1e-5, help="Learning rate for backbone")

    return parser.parse_args()

def main(args):
    device = torch.device(args.device)

    # データセットとデータローダーの準備
    dataset_val = build_dataset(image_set="val", args=args)
    data_loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, collate_fn=utils.collate_fn)

    # モデルの構築
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    # チェックポイントファイルの読み込み
    checkpoint_path = Path(args.output_dir) / args.model_checkpoint
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # モデル全体ではなくレイヤー単位で保存されているため、直接ロード
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded model checkpoint from {checkpoint_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # 評価の実行
    print("Starting evaluation...")
    coco = get_coco_api_from_dataset(dataset_val)
    coco_evaluator = evaluate(model, criterion, postprocessors, data_loader_val, coco, device, output_dir=args.output_dir)
    
    # IoU=0.3を含む評価を設定
    if hasattr(coco_evaluator, "coco_eval") and "bbox" in coco_evaluator.coco_eval:
        iouThrs = coco_evaluator.coco_eval["bbox"].params.iouThrs
        if 0.3 not in iouThrs:
            new_iouThrs = torch.cat([iouThrs, torch.tensor([0.3])])  # 0.3を追加
            coco_evaluator.coco_eval["bbox"].params.iouThrs = new_iouThrs.sort()[0]  # 昇順でソート
    else:
        print("coco_evaluator does not have coco_eval attribute with bbox. Evaluation might be incomplete.")

    # 評価結果の表示
    if hasattr(coco_evaluator, "coco_eval") and "bbox" in coco_evaluator.coco_eval:
        stats = coco_evaluator.coco_eval["bbox"].stats
        print("Evaluation results (COCO bbox metrics):")
        print(f"Average Precision (AP) @[ IoU=0.30      | area=   all | maxDets=100 ] = {stats[0]:.3f}")  # IoU=0.3
        print(f"Average Precision (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[1]:.3f}")
        print(f"Average Precision (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = {stats[2]:.3f}")
        print(f"Average Precision (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = {stats[3]:.3f}")
        print(f"Average Recall (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {stats[8]:.3f}")
    else:
        print("Error: COCO evaluation metrics could not be displayed.")

if __name__ == "__main__":
    args = get_args()
    main(args)
