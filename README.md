# AMED
## 目次
- [概要](#概要)
- [公式実装からの変更点](#公式実装からの変更点)
- [使用環境](#使用環境)
- [トレーニング](#トレーニング)
- [コマンドライン引数](#コマンドライン引数の意味)
- [MAE](#mae)
- [評価](#評価)
- [自己教師あり学習](#自己教師あり学習)

## 概要
[DETR](https://github.com/facebookresearch/detr)のCNNベースのbackboneをMAEを用いたbackboneに変更することを考える．[MAE](https://github.com/facebookresearch/mae)は画像分類で高精度を記録している手法で，その特徴抽出機構が物体検出に良い影響を与えると考えた．また，他の手法に比べDETRはbackbone以降の処理にtransformerを用いており，実装の難易度が低いと考えた．

このコードでの精度評価に関しては[国立研究開発法人日本医療研究開発法人(AMED)](https://www.amed.go.jp)から提供されたデータセットを使用する．

## 公式実装からの変更点
このコードを書く際の目的はDETRのbackboneにMAEを適応させること．主に`./models/backbone.py`を変更．その他コードは適宜用途により調整を行った．

[backborn.py](https://github.com/batumaru12/AMED/blob/main/models/backbone.py)について詳しく説明する．ViTBackbornは本来CNNベースのバックボーンを使っているDETRにViTベースのバックボーンを採用したものである．ViTMAEBackborneはViTベースのMAEから出力された事前学習モデルに適応したバックボーンである．

MAEによる事前学習結果を用いて学習する場合は，ViTMAEBackborneを使用することになる．

また，検出結果を確認した結果一つの物体に対する検出枠が複数得られることが多かった．よってMNSの実装を行った．MNSは[detr.py](https://github.com/batumaru12/AMED/blob/main/models/detr.py)に実装されている．285行目あたりに書かれたuse_nmsがNMSの使用フラグとなっており，TrueにすることでNMSを使用する．

![フローチャート](./fig/proposed_method.jpeg)

## 使用環境
- python 3.12.10
- pytorch 2.5.1 [ダウンロードコマンド](https://pytorch.org/get-started/previous-versions/)
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118
```
- cython 3.0.11
```
pip install cython==3.0.11
```
- scipy 1.13.1
```
pip install scipy==1.13.1
```
pycocotools 2.0.8
```
pip install pycocotools==2.0.8
```
- timm 1.0.11
```
pip install timm==1.0.11
```
- matplotlib 3.9.2
```
pip install matplotlib
```
- transformers 4.45.2
```
pip install transformers==4.45.2
```

venvにて仮想環境を構築．

## トレーニング
学習を行うには[train.py](https://github.com/batumaru12/AMED/blob/main/train.py)を使用する．大本は[DETRの公式実装](https://github.com/facebookresearch/detr)の[main.py](https://github.com/facebookresearch/detr/blob/main/main.py)を参考に作成した．変更点はそれぞれのエポックごとに学習結果を出力するように変更した．モデルをエポックごとに保存する分，PCの容量を多く消費するので注意すること．また，提案手法を実現するために必要なコマンドライン引数の追加を行った．

エポック 500 バッチサイズ 16 でトレーニング:
```
python train.py --batch_size 16 --epochs 500 --lr_drop 350 --num_classes 2 --num_queries 10
```
`--num_classes`はデータセットのクラス数+1にすること．分類するクラス数+背景クラスのため．

MAEを用いて学習する場合:
```
python train.py --batch_size 16 --epochs 500 --lr_drop 350 --num_classes 2 --backbone usemae --num_queries 10 --mae_weights_path MAEの事前学習結果(.pthファイル)
```

複数GPUを使って学習を行う場合:
```
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 --use_env
```
これを`train.py`の前につける．また，エラーが出るため`--find_unused_parameters`をつけること．

### コマンドライン引数の意味
- `--lr` 学習率の設定(デフォルト: 1e-4)
- `--batch_size` バッチサイズの設定　使用環境に合わせて変更(デフォルト: 2)
- `--epochs` エポック数の設定　公式の実験では最終的に500エポック(デフォルト: 300)
- `--lr_drop` 学習率を減衰させるエポック数　デフォルトと同じ割合に設定すればよい(デフォルト: 200)
- `--num_classes` 部隊検出のクラス数　データセットのクラス数+1(背景クラス)に設定(デフォルト: None)
- `--backbone` バックボーンの種類を設定　vitでViTバックボーン，maevitでMAEバックボーン(デフォルト: resnet50)
- `--num_queries` クエリ数を設定　最大検出枠+10ぐらいに設定(デフォルト: 100)
- `--coco_path` cocoデータセットが入ったフォルダを指定(デフォルト: ./coco)
- `--output_dir` 結果とログの出力フォルダを設定
- `--device` CPUを使うかGPUを使うか(デフォルト: cuda)
- `--resume` DETRの事前学習済みモデルを設定
- `--start_epoch` 途中から学習を再開する場合，そのエポックを設定(デフォルト: 0)
- `--mae_weights_path` MAEの事前学習済み重みを設定
- `--mae_mask_ratio` MAEのマスク率を設定(デフォルト: 0.75)

学習の進行状況を`--output_dir`に設定したパスに`log.txt`として保存される．[plot.py](https://github.com/batumaru12/AMED/blob/main/plot.py)を使用することで，グラフにすることが可能．学習状況の確認に適宜利用すること．

## MAE
バックボーンにMAEを用いる場合，[Masked Autoencoderの公式実装](https://github.com/facebookresearch/detr/blob/main/main.py)による事前学習が必要．DETRはcoco形式のデータセットを使うが，MAEはcoco形式のデータセットに対応していない．coco形式のデータセットに対応させたMAEを[]()に置く．これを使て学習してできた.pthファイルを`--mae_weights_path`に指定すること．

## 評価
評価は[eval.py](https://github.com/batumaru12/AMED/blob/main/eval.py)を使う．pycocotoolsを用いたAPとARの算出が可能．`--dataset_file`で入力されたフォルダの画像を対象に評価を行う．

可視化された結果を得たい場合は[detection_result.py](https://github.com/batumaru12/AMED/blob/main/detection_result.py)を使う．`--image_dir`内すべての画像を対象に推論結果の可視化を行い，`--outout_dir`に出力．

## 精度評価
評価方法として先行研究(YOLOv3)と，ViTをそのまま事前学習したもの，ViTをMAEを用いて事前学習したもので比較した．評価する際のIoU閾値は0.1とする(先行研究のYOLOv3の評価に合わせた)．

|model|AP|AR|F1 score|dataset|
|----|:----:|:----:|:----:|:----:|
|YOLOv3|0.873|0.885|0.879|[AMED](https://www.amed.go.jp)|
|DETR(ViT)|0.919|0.977|0.944|[AMED](https://www.amed.go.jp)|
|DETR(MAE)|0.919|0.980|0.949|[AMED](https://www.amed.go.jp)|

先行研究のYOLOv3に比べ，いずれの手法でも高い結果が得られた．今後は，閾値を高くした場合に精度が落ちにくいモデルの実装が必要となる．

## 自己教師あり学習
本研究では，与えられたデータセットではアノテーションされていない腫瘍があることを前提に，一度目の検出で過剰に腫瘍を検出し，その結果から疑似ラベルを得て再び学習を行う．検出結果から疑似ラベルを得るために，[detection_result.py](https://github.com/batumaru12/AMED/blob/main/detection_result.py)で`--save_json`を指定する．`--save_json`を設定することで検出結果を画像で出力するのと同時に，coco形式のjsonファイルで検出結果を保存する．