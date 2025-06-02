import json
import os

def load_meta_filenames(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return {
        os.path.basename(line.strip().split()[0])
        for line in lines if "/meta/" in line
    }

def load_coco_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def build_image_id_map(images):
    return {img["file_name"]: img["id"] for img in images}

def replace_meta_annotations(meta_filenames, json_a, json_b):
    from copy import deepcopy

    # file_name -> image_id マップ作成
    image_file_map_a = {img["file_name"]: img["id"] for img in json_a["images"]}
    image_file_map_b = {img["file_name"]: img["id"] for img in json_b["images"]}

    # image_id -> file_name の逆引きも作る（A）
    image_id_to_file_a = {v: k for k, v in image_file_map_a.items()}

    # meta画像だけ対象にする
    meta_filenames = set(meta_filenames)

    # 現在の最大ID（新しいアノテーションIDのため）
    max_id = max(ann["id"] for ann in json_a["annotations"])

    replaced_annotations = []

    for file_name in meta_filenames:
        image_id_a = image_file_map_a.get(file_name)
        image_id_b = image_file_map_b.get(file_name)

        if image_id_a is None or image_id_b is None:
            continue  # ファイルがどちらかに存在しない

        # Aから元のアノテーション（1つのみ前提）
        anns_a = [ann for ann in json_a["annotations"] if ann["image_id"] == image_id_a]
        if not anns_a:
            continue

        base_ann = deepcopy(anns_a[0])
        base_id = base_ann["id"]

        # Bからアノテーション（複数あるかもしれない）
        anns_b = [ann for ann in json_b["annotations"] if ann["image_id"] == image_id_b]
        if not anns_b:
            replaced_annotations.append(base_ann)
            continue

        # 最初のアノテーションは A の ID を使用
        ann_b0 = anns_b[0]
        for key in ["bbox", "segmentation", "area", "iscrowd"]:
            base_ann[key] = deepcopy(ann_b0.get(key, base_ann.get(key)))
        replaced_annotations.append(base_ann)

        # 2番目以降のBアノテーションを新規追加（新ID割り当て）
        for ann_b in anns_b[1:]:
            max_id += 1
            new_ann = {
                "id": max_id,
                "image_id": image_id_a,  # Aのimage_idを使う
                "category_id": base_ann["category_id"],
                "bbox": deepcopy(ann_b.get("bbox", [])),
                "segmentation": deepcopy(ann_b.get("segmentation", [])),
                "area": ann_b.get("area", 0),
                "iscrowd": ann_b.get("iscrowd", 0)
            }
            replaced_annotations.append(new_ann)

    # meta以外のアノテーション（image_id経由でfile_name判定）
    untouched_annotations = [
        deepcopy(ann)
        for ann in json_a["annotations"]
        if image_id_to_file_a.get(ann["image_id"]) not in meta_filenames
    ]

    # ソートしてマージ
    merged_json = deepcopy(json_a)
    merged_json["annotations"] = sorted(
        untouched_annotations + replaced_annotations,
        key=lambda ann: ann["id"]
    )

    return merged_json

def save_coco_json(coco_json, output_path):
    with open(output_path, 'w') as f:
        json.dump(coco_json, f)

if __name__ == "__main__":
    # パスを指定
    test_txt_path = "amed/annotations/val.txt"
    json_a_path = "amed/save_json_folder/original_annotation_data/instances_val2017.json"
    json_b_path = "amed/save_json_folder/new_data_mae/instances_val2017.json"
    output_path = "instances_val2017.json"

    # 実行
    meta_filenames = load_meta_filenames(test_txt_path)
    json_a = load_coco_json(json_a_path)
    json_b = load_coco_json(json_b_path)
    merged = replace_meta_annotations(meta_filenames, json_a, json_b)
    save_coco_json(merged, output_path)

    print(f"✅ 統合ファイルを出力しました: {output_path}")
