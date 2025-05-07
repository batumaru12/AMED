import json
import matplotlib.pyplot as plt
import os

log_file_path = "./amed/train_weights/new_data_mae/log.txt" #結果を保存したディレクトリのlog.txtを参照
output_dir = "./amed/output/new_data_plot/" #プロット結果を出力するフォルダ

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

epochs = []
train_lr = []
train_loss = []
test_loss = []
train_loss_bbox = []
test_loss_bbox = []
train_loss_giou = []
test_loss_giou = []
test_coco_eval_bbox = []

with open(log_file_path, 'r') as log_file:
    for line in log_file:
        try:
            data = json.loads(line.strip())
            epochs.append(data.get("epoch", len(epochs)))  # エポック番号
            train_lr.append(data.get("train_lr", None))
            train_loss.append(data.get("train_loss", None))
            test_loss.append(data.get("test_loss", None))
            train_loss_bbox.append(data.get("train_loss_bbox", None))
            test_loss_bbox.append(data.get("test_loss_bbox", None))
            train_loss_giou.append(data.get("train_loss_giou", None))
            test_loss_giou.append(data.get("test_loss_giou", None))
            test_coco_eval_bbox.append(data.get("test_coco_eval_bbox", [None])[0])  # 適合率・再現率の例
        except json.JSONDecodeError:
            continue

def save_plot(x, y, xlabel, ylabel, title, filename):
    """指定されたデータをプロットしてPNGファイルに保存"""
    plt.figure()
    # plt.plot(x, y, marker="o") #プロット点あり
    plt.plot(x, y) #プロット点無し
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

save_plot(epochs, train_lr, "Epochs", "train_lr", "train learning rate", output_dir + "use_mae_train_lr.png") 

save_plot(epochs, train_loss, "Epochs", "Loss", "Training Loss", output_dir + "use_mae_train_loss.png")

save_plot(epochs, test_loss, "Epochs", "Loss", "Test Loss", output_dir + "use_mae_test_loss.png")

save_plot(epochs, train_loss_bbox, "Epochs", "BBox Loss", "Training Bounding Box Loss", output_dir + "use_mae_train_bbox_loss.png")
save_plot(epochs, test_loss_bbox, "Epochs", "BBox Loss", "Test Bounding Box Loss", output_dir + "use_mae_test_bbox_loss.png")

save_plot(epochs, train_loss_giou, "Epochs", "GIoU Loss", "Training GIoU Loss", output_dir + "use_mae_train_giou_loss.png")
save_plot(epochs, test_loss_giou, "Epochs", "GIoU Loss", "Test GIoU Loss", output_dir + "use_mae_test_giou_loss.png")

save_plot(epochs, test_coco_eval_bbox, "Epochs", "COCO Eval (Precision/Recall)", "COCO Evaluation Metrics", output_dir + "use_mae_coco_eval.png")
