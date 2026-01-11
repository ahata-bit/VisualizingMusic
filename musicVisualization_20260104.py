import librosa
import numpy as np
from collections import Counter
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import random
from math import cos, sin, sqrt, pi
from PIL import Image, ImageGrab, EpsImagePlugin
from scipy.ndimage import median_filter


#テンポ推定-----------------------------------------------
file_path = input("解析したい音声ファイルのパスを入力してください: ")

# 音声ファイルを読み込み
y_1, sr_1 = librosa.load(file_path, sr=22050)

# BPM（テンポ）推定
onset_env = librosa.onset.onset_strength(y=y_1, sr=sr_1)
tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_1)

# tempoが配列の場合は最初の要素を使う
if isinstance(tempo, (list, tuple, np.ndarray)):
    bpm = tempo[0]
else:
    bpm = tempo
#--------------------------------------------------------

#ジャンル推定---------------------------------------------
# 事前にモデルをロード
music2vec = tf.keras.models.load_model('E:/zemi/GC/music2vec_10epochs.keras')

# ジャンルリスト（学習時と同じ順番で）
genre_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
color_list = [
    (255 , 173 , 173),    # blues
    (255 , 173 , 214),    # classical 
    (255 , 173 , 255),    # country
    (214 , 173 , 255),  # disco
    (173 , 173 , 255),  # hiphop
    (173 , 214 , 255),  # jazz
    (173 , 255 , 255),# metal
    (173 , 255 , 214),  # pop
    (173 , 255 , 173),   # reggae
    (214 , 255 , 173) # rock
]

# 前処理
y, sr = librosa.load(file_path, sr=22050, mono=True, dtype=np.float32)
window_size = 675808
step = window_size

results = []
for start in range(0, len(y) - window_size + 1, step):
    chunk = y[start:start+window_size]
    chunk = chunk * 256.0
    chunk = np.reshape(chunk, (-1, 1))
    input_data = np.expand_dims(chunk, axis=0)  # (1, 675808, 1)
    pred = music2vec.predict(input_data)
    results.append(np.argmax(pred))

remainder = len(y) % window_size
if remainder != 0 and len(y) > window_size // 2:
    chunk = y[-remainder:]
    # 0でパディングして長さをwindow_sizeに揃える
    chunk = np.pad(chunk, (0, window_size - remainder))
    chunk = chunk * 256.0
    chunk = np.reshape(chunk, (-1, 1))
    input_data = np.expand_dims(chunk, axis=0)
    pred = music2vec.predict(input_data)
    results.append(np.argmax(pred))

# 多数決で最終ジャンル決定
if results:
    predicted_label = Counter(results).most_common(1)[0][0]
else:
    print("エラー: 推論結果がありません。音声ファイルやwindow_sizeを確認してください。")
#---------------------------------------------------------

#マーブリング---------------------------------------------------
y, sr = librosa.load(file_path, sr=22050, duration=60)  # duration=60で最初の1分のみ

f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
#-f0 = f0[~np.isnan(f0)]
midi_notes = librosa.hz_to_midi(f0)
# --- F0スムージング：ノイズを除去してロングトーンの誤検出を防ぐ ---
# まず有声音フレームのみを抽出（voiced_probs > 0.9）
voiced_mask = voiced_probs > 0.9
f0_voiced = f0[voiced_mask]
f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

# メディアンフィルタでノイズ除去（ウィンドウ幅5フレーム）
f0_smoothed = median_filter(f0_voiced, size=5)

# MIDI に変換
midi_notes = librosa.hz_to_midi(f0_smoothed)
# --- F0スムージング終了 ---

color_map_rgb = {
    0: (255, 0, 0),    # ユニゾン (白) 
    1: (255, 69, 0),    # 短2度 (オレンジレッド)
    2: (255, 140, 0),   # 長2度 (ダークオレンジ)
    3: (255, 215, 0),   # 短3度 (ゴールド)
    4: (173, 255, 47),  # 長3度 (イエローグリーン)
    5: (0, 255, 0),     # 完全4度 (ライムグリーン)
    6: (0, 206, 209),   # 増4度/減5度 (ダークターコイズ)
    7: (0, 0, 255),     # 完全5度 (青)
    8: (75, 0, 130),    # 短6度 (インディゴ)
    9: (148, 0, 211),   # 長6度 (ダークバイオレット)
    10: (238, 130, 238),# 短7度 (バイオレット)
    11: (255, 0, 255)   # 長7度 (マゼンタ)
}

diffs_colors_rgb = []
for i in range(len(midi_notes)-1):
    diff = int(round(midi_notes[i+1] - midi_notes[i])) % 12
    # 音高差が±0.5未満（50cents未満）の場合は同じ音と扱う
    diff_exact = midi_notes[i+1] - midi_notes[i]
    if abs(diff_exact) < 0.5:
        diff = 0  # ロングトーン（同じ音）
    else:
        diff = int(round(diff_exact)) % 12
    color = color_map_rgb[diff]
    diffs_colors_rgb.append(color)
    print(f"{i}番目: 音高差 {diff} -> 色 (RGB): {color}")

# --- ここからTkinter描画部 ---
window_width = 720
window_height = 720

def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb

root = tk.Tk()


def save_canvas():
    file_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        title="画像の保存先とファイル名を指定してください"
    )
    if not file_path:
        return

    # 描画を確実に反映
    root.update()
    canvas.update()

    # Canvasの絶対座標を取得
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    w = x + canvas.winfo_width()
    h = y + canvas.winfo_height()

    # 画面からCanvas部分をキャプチャ
    img = ImageGrab.grab(bbox=(x, y, w, h))
    img.save(file_path)
    print(f"画像を保存しました: {file_path}")
    
root.title("marbling melody")
root.geometry(f"{window_width}x{window_height}")

# ジャンル色で背景を塗る
genre_color = color_list[predicted_label]
canvas = tk.Canvas(root, width=window_width, height=window_height, bg=rgb_to_hex(genre_color), highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)


# マーブリング描画クラスなど（ここは元のまま）
class config:
    polygonSides = 360
    random_color = True
    random_size = False
    fixed_size = 20
    @staticmethod
    def racalc():
        pass

center = (window_width / 2, window_height / 2)
circles = []

class circle:
    def __init__(self, x: float, y: float, color_rgb: tuple):
        config.racalc()
        self.points = [(x, y)]
        self._points = []
        self.position = (x, y)
        self.circle_radius = config.fixed_size
        self._points.append((self.circle_radius, 0))
        self.side = config.polygonSides
        self.angel = 360 / self.side
        self.radian = (self.angel / 180) * pi
        self.color = rgb_to_hex(color_rgb)
        self.calculate_dot()
        circles.append(self)
    def calculate_dot(self):
        x = self._points[-1][0]
        y = self._points[-1][1]
        xn = x * cos(self.radian) - y * sin(self.radian)
        yn = y * cos(self.radian) + x * sin(self.radian)
        point = (xn, yn)
        self._points.append(point)
        self.points.append((xn + self.position[0], yn + self.position[1]))
        if len(self._points) - 1 > self.side:
            return None
        else:
            self.calculate_dot()
    def draw_dot(self):
        canvas.create_polygon(*self.points, fill=self.color, outline="")
    def draw(self):
        self.draw_dot()


def initialize_marbling_with_colors():
    # 描画範囲を中心正方形に限定
    square_size = window_width // 2
    square_left = window_width // 4
    square_top = window_height // 4
    square_right = square_left + square_size
    square_bottom = square_top + square_size

    for idx, color_rgb in enumerate(diffs_colors_rgb):
        if color_rgb is None:
            continue  # ユニゾンはスキップ
        if idx % 10 != 0:
            continue
        min_x = square_left + config.fixed_size
        max_x = square_right - config.fixed_size
        min_y = square_top + config.fixed_size
        max_y = square_bottom - config.fixed_size
        if min_x < max_x and min_y < max_y:
            rand_x = random.randint(min_x, max_x)
            rand_y = random.randint(min_y, max_y)
            manage_window.draw_circle_at_position(rand_x, rand_y, color_rgb)
        else:
            print("警告: ウィンドウサイズが円の半径に対して小さすぎます。中心に描画します。")
            manage_window.draw_circle_at_position(window_width / 2, window_height / 2, color_rgb)
    
    if bpm > 0:
        marble_break_count = int(len(midi_notes) / bpm)
        if marble_break_count == 1:
            x_positions = [window_width // 2]
        elif marble_break_count == 2:
            x_positions = [window_width // 4, window_width * 3 // 4]
        else:
            x_positions = [int(window_width * (i + 1) / (marble_break_count + 1)) for i in range(marble_break_count)]

        break_width = window_width // 5
        break_range = config.fixed_size * 2
        for x_break in x_positions:
            # 伝播用の初期シフト値
            shift = random.randint(-break_width, break_width)
            # 伝播度合い（0.5なら前のずれの半分を次に伝える）
            propagate = 0.5
        
            # 円をy座標順にソート（上から下へ伝播させるイメージ）
            sorted_circles = sorted(circles, key=lambda c: min(px for (px, y) in c.points))
            prev_shift = shift
            for c in sorted_circles:
                # 円の中心が崩し位置x_break付近ならずらす
                center_x = sum(px for (px, py) in c.points) / len(c.points)
                if abs(center_x - x_break) < break_range:
                    # 前のずれに追従してずらす
                    this_shift = prev_shift * propagate + random.randint(-break_width//10, break_width//10)
                    new_points = [(x + this_shift, py) for (x, py) in c.points]
                    c.points = new_points
                    prev_shift = this_shift
                    manage_window.redraw_circle()

class manage_window:
    @staticmethod
    def mathematical(c: tuple, p: tuple, r: float):
        euc = sqrt(pow((c[0] - p[0]), 2) + pow((c[1] - p[1]), 2))
        right_p = sqrt((pow(r, 2) / pow(euc, 2)) + 1) if euc != 0 else sqrt((pow(r, 2) / 1))
        mines = (p[0] - c[0], p[1] - c[1])
        right = (right_p * mines[0], right_p * mines[1])
        result = (right[0] + c[0], right[1] + c[1])
        return result

    @staticmethod
    def calculate_new_circles(new_circle):
        circle_i = new_circle
        for circle_ii in circles:
            if not circle_ii == circle_i:
                for index in range(0, len(circle_ii.points)):
                    temp = manage_window.mathematical(circle_i.position, circle_ii.points[index], circle_i.circle_radius)
                    circle_ii.points[index] = temp

    @staticmethod
    def draw_circle_at_position(x: float, y: float, color_rgb: tuple):
        new_circle = circle(x, y, color_rgb)
        manage_window.calculate_new_circles(new_circle)
        manage_window.redraw_circle()


    @staticmethod
    def redraw_circle():
        canvas.create_rectangle(0, 0, window_width, window_height, fill=rgb_to_hex(genre_color), outline="")
        for circle_i in circles:
            circle_i.draw()
            
def open_save_window():
    save_win = tk.Toplevel(root)
    save_win.title("画像保存")
    save_win.geometry("200x100")
    btn = tk.Button(save_win, text="画像を保存", command=save_canvas)
    btn.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

# --- 修正：マーブリング描画を root.after() で遅延実行 ---
def draw_marbling_and_show_save():
    initialize_marbling_with_colors()
    # ジャンル名とBPMを描画
    try:
        canvas.create_text(10, 10, text=f"{genre_list[predicted_label]}  BPM:{bpm:.2f}", fill="white", font=("Arial", 20, "bold"), anchor="nw")
    except Exception:
        pass
    manage_window.redraw_circle()
    root.after(500, open_save_window)

# イベントループ開始後に描画を実行
root.after(100, draw_marbling_and_show_save)
root.mainloop()

