# TensorFlowによる音楽ジャンルの分類学習
## 目次
#### 1.使用言語
#### 2.使用ライブラリ
#### 3.実行環境
#### 4.データセット
#### 5.構成
#### 6.実行方法
#### 7.学習モデル構成
#### 8.制作理由
#### 9.課題
#### 10.今後の展望
## 1.使用言語
Python3
## 2.使用ライブラリ
・Numpy 1.24.3<br>
・librosa 0.10.1<br>
・matplotlib 3.6.2<br>
・TensorFlow 2.13.1
## 3.実行環境
ubuntu20.04.6<br>
WSL2<br>
VSCode
## 4.データセット
### 学習用データセット
#### GTZAN Dataset - Music Genre Classification
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
### 推定用データ(Original)
#### 推定用データ　CMSL クラシック名曲サウンドライブラリー【ライセンスフリー素材音源 700曲】
http://classical-sound.seesaa.net/article/474154731.html
## 5.構成
### music_main
メインループ<br>
実行ファイル
### music_functions.py
Windowsで実行する関数群
### music_functions_wsl.py
wslで実行する関数群
### music_trainer.py
データセットの読み込み<br>
モデルの構築、学習
### music_prediction.py
ジャンル推定
## 6.実行方法
### ファイル構造
#### current
music_main.py<br>
music_functions.py<br>
music_functions_wsl.py<br>
music_trainer.py<br>
music_prediction.py<br>
archive<br>
log.txt<br>
※music_trainer.py実行後は、<br>
mfccs_dataset.npz<br>
mfccs_dataset_added.npz<br>
mfcc_model<br>
mfcc_model_added<br>
music_prediction.py実行後は、<br>
predict.png<br>
が追加される
#### archive（データセット）
archive/genres/各ジャンルフォルダー/各ジャンル100曲のデータ
### 実行手順
music_mainを実行
## 7.学習モデル構成
### conv1d
活性化関数 = ReLu<br>
ノード数 = 256
### max_pooling1d
### conv1d
活性化関数 = ReLu<br>
ノード数 = 128<br>
### max_pooling1d
### Flatten
### Dense
活性化関数 = ReLu<br>
ノード数 = 256<br>
### Dropout
### Dense
### optimizer = Adam
### epoch = 10
## 8.制作理由
大学で機械学習について学習していくうえで深層学習に興味を持つ。<br>
趣味のDTMと関連付けて何かできないかと考え、音楽ジャンルの分類に挑戦したいと考え制作。

## 9.課題
・モデル構築にまだ考える余地あり<br>
・特徴量をMFCC変換以外の手法も考慮する必要がある<br>
・推定対象の曲が長尺のときの対応<br>
・WSLでGUIが対応しておらず、WindowsがTensorFlowの最新版に対応していないためかなり複雑な構成となったため、環境の見直し
## 10.今後の展望
・推定結果に合わせた音声処理<br>
・exe形式のアプリケーション化、VSTプラグイン化による実用化
