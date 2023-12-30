import librosa
import numpy as np
import sys
import os

# 音楽ファイル読み込み
def load_data(file_path, min_frames,max_fs):
    # データ読み込み
    music, fs = librosa.load(file_path)

    # MFCC変換
    mfcc = librosa.feature.mfcc(y=music, sr=max_fs, n_mfcc=30)

    # numpy配列化
    mfcc = np.array(mfcc)

    # 推定対象データがmin_framesよりも大きい場合
    if mfcc.shape[1] > min_frames:
            
        mfcc = [np.resize(mfcc,(30,min_frames))]
        mfcc = np.array(mfcc)

    # 推定対象データがmin_framesよりも小さい場合
    else:
        # 配列をmin_framesに合わせて0埋め
        mfcc = [np.pad(mfcc, ((0, 0), (0, max(0, min_frames - mfcc.shape[1]))), mode='constant')]        
        
    return mfcc

# データセットへ書き込み
def add_data_append(file_path, label):

    # mfccs_dataset_addedがすでにある場合
    if os.path.exists('../system/datas/mfccs_dataset_added.npz'):
        data = np.load('../system/datas/mfccs_dataset_added.npz', allow_pickle=True)
    # オリジナルデータしかない場合
    else:
        data = np.load('../system/datas/mfccs_dataset.npz', allow_pickle=True)

    # data展開
    mfccs = np.array(data['arr_0'])
    labels = data['arr_1']
    max_fs = data['arr_2']
    min_frames = min(len(mfcc[0]) for mfcc in mfccs)

    # 追加データ読み込み
    mfcc = load_data(file_path, min_frames,max_fs)

    # データ追加
    label = np.array([label])
    mfccs = np.concatenate((mfccs,mfcc),axis=0)
    labels = np.concatenate((labels,label),axis=0)

    # 保存
    try:
        mfccs = np.array(mfccs)

        # オリジナルデータとは別で保存
        np.savez('../system/datas/mfccs_dataset_added',mfccs,labels,max_fs)
    except Exception as e:
        print('SaveErr : Continue running')

# 入力が3つ（データ追加の時のみ）実行
if len(sys.argv) == 3:
    add_data_append(sys.argv[1],sys.argv[2])