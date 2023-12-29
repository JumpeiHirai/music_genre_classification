# メインループ　GUI表示

import tkinter as tk
from music_functions import *

# ウィンドウ作成
root = tk.Tk()
root.title("音楽ジャンル推定")

# ボタン1 推定
button1 = tk.Button(root, text="ジャンルを推定", command=predict_view)
button1.pack(padx=40, pady=20)

# ボタン2 データ追加
button2 = tk.Button(root, text="モデルデータ追加", command=add_data)
button2.pack(padx=80, pady=20)

# ボタン3 モデル再構築
button3 = tk.Button(root,text="モデル再構築",command=model_reconstruction)
button3.pack(padx=120, pady=20)

root.mainloop()
