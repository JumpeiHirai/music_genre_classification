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
from music_trainer import *
from music_functions_wsl import load_data

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

# 推定
def predict_genre(file_path):
    # ジャンル名抽出のため
    sys.stdout = open(os.devnull, 'w')

    # データセット読み込み
    data = np.load('mfccs_dataset.npz', allow_pickle=True)
    mfccs = np.array(data['arr_0'])
    labels = data['arr_1']
    fs = data['arr_2']

    # 最短時間算出
    min_frames = min(len(mfcc[0]) for mfcc in mfccs)

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
    #model.summary()

    # 正解率算出、表示
    test_loss, test_acc = model.evaluate(np.array(mfccs), np.array(labels_index), verbose = 0)
    #print('\nTest accuracy = ',test_acc)

    # 推定用モデル構築
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    # ジャンル推定
    mfcc = load_data(file_path, min_frames,fs)

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
    sys.stdout = sys.__stdout__
    return(class_names[np.argmax(predictions[0])])

# 入力が2つの時（推定時）実行
if len(sys.argv) == 2:
    if not os.path.exists('mfcc_model'):
        if not os.path.exists('mfccs_dataset.npz'):
            import_data()
        trainer()
    print(predict_genre(sys.argv[1]))

# モデル再構築
else:
    trainer()

