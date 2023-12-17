# TensorFlowによる音楽ジャンルの分類学習
## 使用言語
Python3
## 使用ライブラリ
・Numpy<br>
・librosa<br>
・matplotlib<br>
・TensorFlow
・os
## 構成
### music_trainer.py
データセットの読み込み<br>
モデルの構築、学習
### music_prediction.py
ジャンル推定
## 実行方法
### ファイル構造
#### current
music_trainer.py<br>
music_prediction.py<br>
test_data<br>
archive<br>
※music_trainer.py実行後は、<br>
mfccs_dataset.npz<br>
mfcc_model<br>
music_prediction.py実行後は、<br>
predict.png<br>
が追加される
#### archive（データセット）
archive/genres/各ジャンルフォルダー/各ジャンル100曲のデータ
#### test_data（推定データ）
test_data/Tchaikovsky-Symphony-No5-2nd-2020-AR.mp3
### 実行手順
music_trainer.py実行後、music_prediction.pyを実行


