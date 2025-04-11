# AMED
## 目次
- [概要](#概要)
- [公式実装からの変更点](#公式実装からの変更点)
- [使用環境](#使用環境)
- [トレーニング](#トレーニング)
- [コマンドライン引数](#コマンドライン引数の意味)
- [MAE](#mae)
- [評価](#評価)

## 概要
[DETR](https://github.com/facebookresearch/detr)のCNNベースのbackboneをMAEを用いたbackboneに変更することを考える．[MAE](https://github.com/facebookresearch/mae)は画像分類で高精度を記録している手法で，その特徴抽出機構が物体検出に良い影響を与えると考えた．また，他の手法に比べDETRはbackbone以降の処理にtransformerを用いており，実装の難易度が低いと考えた．

MAEは画像分類の事前額手法であるため，画像特徴は得られるが位置情報が欠落する．DETRでは，backboneの処理後に位置エンコーディングを行っており，その機構を利用した．

## 公式実装からの変更点
このコードを書く際の目的はDETRのbackboneにMAEを適応させること．主に`./models/backbone.py`を変更．その他コードは適宜用途により調整を行った．

[backborn.py](https://github.com/batumaru12/AMED/blob/main/models/backbone.py)について詳しく説明する．ViTBackbornは本来CNNベースのバックボーンを使っているDETRにViTベースのバックボーンを採用したものである．ViTMAEBackborneはViTベースのMAEから出力された事前学習モデルに適応したバックボーンである．

MAEによる事前学習結果を用いて学習する場合は，ViTMAEBackborneを使用することになる．

## 使用環境
- python 3.12.7
- pytorch 2.5.1 [ダウンロードコマンド](https://pytorch.org/get-started/previous-versions/)
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
- cython 3.0.11
- scipy 1.13.1
```
conda install cython scipy
```
pycocotools 2.0.8
```
conda install conda-forge::pycocotools
```
- tqdm 4.66.5
```
conda install tqdm
```
- timm 1.0.11
```
conda install conda-forge::timm
```
- matplotlib 3.9.2
```
conda install matplotlib
```
- transformers 4.45.2
```
conda install transformers
```

Anacondaではない場合はpipなどでインストールしてください．

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
評価方法としてViTをそのまま事前学習したもの，ViTをMAEを用いて事前学習したもので比較した．