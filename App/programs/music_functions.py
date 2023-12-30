# windowsで実行可能な関数

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import subprocess
import numpy as np
import os
from datetime import datetime

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

# add_dataで使用　wslコマンド実行
def wsl_btn(file_path,x):
    wsl_command = f"wsl python3 ./music_functions_wsl.py {file_path} {x}"
    subprocess.Popen(wsl_command, shell=True)
    
# データ追加
# 入力があるとき(推定後)、入力がないとき(データ追加)に対応
def add_data(arg1=0):
    if arg1 == 0:
        file_path = filedialog.askopenfilename()
        file_path = convert_to_wsl_path(file_path)
    else:
        file_path = arg1
    # GUI
    root = tk.Tk()
    x = tk.StringVar()
    frame = tk.Frame(root)
    combo = ttk.Combobox(frame,textvariable = x,values=class_names,width=10)
    combo.set(class_names[0])
    combo.grid()

    # コンボボックスの値取得
    x = combo.get()

    # ボタン　データ追加　wslコマンド実行
    button = tk.Button(frame,text='OK',command = lambda: (wsl_btn(file_path,x),root.destroy()))
    button.grid()
    frame.grid()

    # ログファイル入力
    log = open('../system/log.txt','a')
    log.write(str(datetime.now()) + ': added      : ' + str(file_path) + '' + str(x) + '\n')

# ファイルパスの変換
def convert_to_wsl_path(path):
    print(path)
    command = f"wsl wslpath -u {path}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return result.stdout.strip()

# 推定結果表示
def predict_view():

    # ウィンドウ生成
    root = tk.Toplevel()

    # 推定結果テキスト
    genre_name, file_path = predict()
    result_text = tk.StringVar()
    result_text.set(f"この曲は {genre_name} ですね！")
    result_label = tk.Label(root, textvariable=result_text,font=("Menlo", 50))
    result_label.pack()

    # スペクトラム表示
    img = Image.open("../system/predict.png")
    img = ImageTk.PhotoImage(img)
    image_label = tk.Label(root,image=img)
    image_label.image = img
    image_label.pack()

    # ログファイル入力
    log = open('../system/log.txt','a')
    log.write(str(datetime.now()) + ': predicted  : ' + str(file_path) + ' ' + str(genre_name) + '\n')

    # 推定ファイルのデータセット追加
    result = messagebox.askokcancel("", "実行ファイルをデータセットへ追加しますか？")
    if result:
        add_data(file_path)

# 推定
def predict():
    # ファイル選択
    file_path = filedialog.askopenfilename()

    # ファイルパス変換
    file_path = convert_to_wsl_path(file_path)

    # 実行コマンド
    wsl_command = f"wsl python3 ./music_prediction.py {file_path}"
        
    # wslコマンド実行+ジャンル名抽出
    subprocess.run(wsl_command, shell=True)
    genre_name = np.load('../system/genre_name.npy')

    return str(genre_name), file_path

# モデル再構築
def model_reconstruction():
    # 警告
    result = messagebox.askokcancel("注意", "実行しますか？")
    if result:
        # 引数なしでmusic_prediction.pyを実行
        wsl_command = f"wsl python3 ./music_prediction.py"
        subprocess.Popen(wsl_command, shell=True)

        # ログファイル入力
        log = open('../system/log.txt','a')
        log.write(str(datetime.now()) + ': modelconst : ' + 'Success' + '\n')