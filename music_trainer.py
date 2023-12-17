# 音楽ジャンル分類モデル構築＆保存

# 学習用データセット　GTZAN Dataset - Music Genre Classification
# https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

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

# mfcc_dataset.npz(データセット)がない場合、データを読み込み
# ./archive/genres/各ジャンル/データ
if os.path.exists('mfccs_dataset.npz') == False:
    mfccs = []
    labels = []
    path = './archive/genres'
    max_frames = 0
    max_fs = 0
    for foldername in os.listdir(path):
        for filename in os.listdir(path + '/' + foldername):
            # 読み込み
            try:
                music, fs = librosa.load(path + '/' + foldername + '/' + filename)
                print('load file : ',filename)
                # MFCC変換
                mfcc = librosa.feature.mfcc(y=music, sr=fs, n_mfcc=20)

                # リストに追加
                mfccs.append(mfcc)
                labels.append(foldername)

                # サンプリングレート、時間の最大値を保持
                max_fs = max(max_fs, fs)
                max_frames = max(max_frames, mfcc.shape[1])

            except OSError as e:
                print('Not Found : ' + filename)
    
    # 最長の曲に合わせて0埋め
    mfccs = [np.pad(mfcc, ((0, 0), (0, max_frames - mfcc.shape[1])), mode='constant') for mfcc in mfccs]
    # mfcc変換データ, ラベル, サンプリングレートを保存
    try:
        np.savez(os.path.join('./', 'mfccs_dataset.npz'),mfccs,labels,max_fs)
    except Exception as e:
        print('SaveErr : Continue running')

# mfcc_dataset.npzがある場合、データセットの読み込み
else:
    data = np.load('mfccs_dataset.npz', allow_pickle=True)
    mfccs = data['arr_0']
    labels = data['arr_1']

# mfccsの正規化(Z-score)
mean = np.mean(mfccs, axis = (0,2),keepdims=True)
std = np.std(mfccs,axis = (0,2),keepdims=True)
mfccs = (mfccs - mean) / std 

# ラベルのインデックス化(0～9)
labels_index = []
for label in labels:
    labels_index.append(class_names.index(label))

# データセット化、データセットのシャッフル
dataset = tf.data.Dataset.from_tensor_slices((mfccs,labels_index))
dataset = dataset.shuffle(buffer_size=len(mfccs) * 3)

# 訓練、検証、テストデータに分割
train_split=0.8
val_split=0.1
test_split=0.1
train_size = int(train_split * mfccs.shape[0])
val_size = int(val_split * mfccs.shape[0])
train_mfccs = dataset.take(train_size)    
val_mfccs = dataset.skip(train_size).take(val_size)
test_mfccs = dataset.skip(train_size).skip(val_size)

# 訓練データをmfccとラベルに分割
train_mfcc = []
train_label = []
for mfcc, label in train_mfccs:
    train_mfcc.append(mfcc)
    train_label.append(label)

# テストデータをmfccとラベルに分割
test_mfcc = []
test_label = []
for mfcc, label in test_mfccs:
    test_mfcc.append(mfcc)
    test_label.append(label)

# 検証データをmfccとラベルに分割
val_mfcc = []
val_label = []
for mfcc, label in val_mfccs:
    val_mfcc.append(mfcc)
    val_label.append(label)

# エポック設定
num_epoch = 20

# 学習モデル構築
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(256, 5, activation='relu', input_shape=(mfccs.shape[1], mfccs.shape[2])),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(input_shape=(mfccs[0].shape[0], mfccs[0].shape[1])),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer = 'Adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
    metrics = ['accuracy']
)

# モデルトレーニング
model.fit(np.array(train_mfcc),np.array(train_label), epochs = num_epoch,validation_data=(np.array(val_mfcc), np.array(val_label)))

# テスト
test_loss, test_acc = model.evaluate(np.array(test_mfcc),np.array(test_label), verbose = 2)
print('\nTest accuracy = ', test_acc)

# モデルの保存
try:
    model.save('mfcc_model')
except Exception as e:
    print('SaveErr : Couldn\'t save model')
