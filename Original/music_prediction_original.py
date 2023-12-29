# 音楽ジャンル推定
# music_trainer.py実行後

# 推定用データ　CMSL クラシック名曲サウンドライブラリー【ライセンスフリー素材音源 700曲】
# http://classical-sound.seesaa.net/article/474154731.html

import sys
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import numpy as np

# 10ジャンル
class_names = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]

# データセット読み込み
data = np.load('mfccs_dataset.npz', allow_pickle=True)
mfccs = np.array(data['arr_0'])
labels = data['arr_1']
fs = data['arr_2']

# 最長時間算出
max_frames = max(mfcc.shape[1] for mfcc in mfccs)

# ラベルのインデックス化
labels_index = []
for label in labels:
    labels_index.append(class_names.index(label))

# データセット正規化
mean = np.mean(mfccs, axis=(0,2),keepdims = True)
std = np.std(mfccs, axis=(0,2),keepdims = True)
mfccs = (mfccs - mean) / std

# モデルの読み込み
model = tf.keras.models.load_model('mfcc_model')

# モデル情報参照
model.summary()

# 正解率算出、表示
test_loss, test_acc = model.evaluate(np.array(mfccs), np.array(labels_index), verbose = 2)
print('\nTest accuracy = ',test_acc)

# 推定用モデル構築
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# ジャンル推定
# データ読み込み
music, fs = librosa.load('./test_data/Tchaikovsky-Symphony-No5-2nd-2020-AR.mp3', sr=fs)

# MFCC変換
mfcc = librosa.feature.mfcc(y=music, sr=fs, n_mfcc=20)

# numpy配列化
mfcc = np.array(mfcc)

# 推定対象データがmax_framesよりも小さい場合
if mfcc.shape[1] < max_frames:
    # 配列をmax_framesに合わせて0埋め
    mfcc = [np.pad(mfcc, ((0, 0), (0, max(0, max_frames - mfcc.shape[1]))), mode='constant')]
    mfcc = np.array(mfcc)

# 推定対象データがmax_framesよりも大きい場合
else:
    #データをカッティングしmax_framesに合わせる
    mfcc = librosa.resample(mfcc, orig_sr=mfcc.shape[1], target_sr=1320)

# スペクトラム生成、正規化のため次元変換
# 次元数が3次元でないなら3次元へ変換
if np.ndim(mfcc) != 3:
    mfcc = mfcc.reshape(1,mfcc.shape[0],mfcc.shape[1])

# スペクトラム画像保存
plt.figure()
librosa.display.specshow(mfcc[0], sr=fs, x_axis='time', y_axis='log')
plt.savefig('predict.png')

# 推定対象データの正規化
mfcc = (mfcc - mean) / std

# 次元数を2次元に戻す
mfcc = np.expand_dims(mfcc, axis=-1)

# 推定
predictions = probability_model.predict(mfcc)

# 推定結果表示
print('predictions = ', predictions[0])
print('argmax(predictions) = ', np.argmax(predictions[0]))
print('class_names[argmax(predictions)] = ', class_names[np.argmax(predictions[0])])
