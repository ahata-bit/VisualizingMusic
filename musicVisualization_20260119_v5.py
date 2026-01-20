import librosa
import numpy as np
from collections import Counter
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import random
from math import cos, sin, sqrt, pi, atan2
from PIL import Image, ImageGrab, EpsImagePlugin
from scipy.ndimage import median_filter
import os
import time
import glob

# マーブリング崩し箇所の重み（コード内で変更してください）
MARBLE_BREAK_WEIGHT = 1.0
# 崩しの強さ（大きくすると変位が増える）
MARBLE_BREAK_INTENSITY = 2.0
# 全体的な変位スケール（disp_x/disp_y と swirl に乗算）
DISPLACEMENT_SCALE = 1.0

# --- 追加：複数ファイルの読み込みとテキスト出力 ---
folder_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
data_list = []
output_txt = "loaded_wav_files.txt"
all_wav_paths = []

print("全ジャンルフォルダからwavファイルを読み込み中...")

with open(output_txt, "w", encoding="utf-8") as f:
    for n in folder_names:
        # ディレクトリ構造に合わせてパスを構築（適宜修正してください）
        #folder_path = f'E:/GC/music/Data/genres_original/{n}/'
        folder_path = f'E:\GC\music' 
        # フォルダ内のwavファイルを検索
        wav_files = glob.glob(os.path.join(folder_path, "*.wav"))
        
        for file_path in wav_files:
            try:
                # 1. テキストファイルに書き込み
                f.write(file_path + "\n")
                
                # 2. 音声の読み込み (メモリ節約のため、解析に必要なsr=22050で読み込み)
                y_loaded, sr = librosa.load(file_path, sr=22050)
                data_list.append(y_loaded)
                
                print(f"Loaded: {os.path.basename(file_path)} (Total: {len(data_list)})")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

if not data_list:
    print("エラー: 音声ファイルが一つも読み込まれませんでした。パスを確認してください。")
    exit()

print(f"--- 読み込み完了。合計 {len(data_list)} ファイルをリスト化しました。 ---")

# 解析対象の選択（ここではリストの最初のファイルを解析に使用します）
# 必要に応じて解析したいファイルのインデックスを指定してください
target_index = 0
y_active = data_list[target_index]
sr_active = 22050
file_path = all_wav_paths[target_index] # 既存コードとの互換性のため保持

#--------------------------------------------------------
# テンポ推定（一括読み込みしたデータを使用）
#--------------------------------------------------------
print(f"\n解析中のファイル: {os.path.basename(file_path)}")

# BPM（テンポ）推定
onset_env = librosa.onset.onset_strength(y=y_active, sr=sr_active)
tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr_active)

if isinstance(tempo, (list, tuple, np.ndarray)):
    bpm = tempo[0]
else:
    bpm = tempo
print(f"検出されたBPM: {bpm:.2f}")

#--------------------------------------------------------
# ジャンル推定
#--------------------------------------------------------
music2vec = tf.keras.models.load_model('E:/zemi/GC/music2vec_10epochs.keras')
MODEL_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
VISUAL_GENRE_ORDER = ['metal','rock','pop','country','disco','blues','jazz','reggae','hiphop','classical']
VISUAL_COLOR_LIST = [
    (255,173,173), (255,173,214), (255,173,255), (214,173,255), (173,173,255),
    (173,214,255), (173,255,255), (173,255,214), (173,255,173), (214,255,173)
]
color_map = dict(zip(VISUAL_GENRE_ORDER, VISUAL_COLOR_LIST))

window_size = 675808
step = window_size
results = []

# すでに読み込んである y_active を使用
for start in range(0, len(y_active) - window_size + 1, step):
    end = start + window_size
    chunk = y_active[start:end]
    chunk = chunk * 256.0
    chunk = np.reshape(chunk, (-1, 1))
    input_data = np.expand_dims(chunk, axis=0)
    pred = music2vec.predict(input_data)
    pred_idx = int(np.argmax(pred))
    pred_conf = float(np.max(pred))
    results.append(pred_idx)
    
    start_sec = start / sr_active
    end_sec = end / sr_active
    pred_genre = MODEL_GENRES[pred_idx]
    print(f"Segment {start_sec:.2f}s-{end_sec:.2f}s -> {pred_genre} (idx={pred_idx}, conf={pred_conf:.3f})")

# 多数決で最終ジャンル決定
if results:
    predicted_label = Counter(results).most_common(1)[0][0]
    final_genre = MODEL_GENRES[predicted_label]
    print(f"最終ジャンル: {final_genre}")
else:
    predicted_label = 0
    final_genre = "Unknown"

#---------------------------------------------------------
# マーブリング用のピッチ解析
#---------------------------------------------------------
# 最初の1分のみ使用
y_short = y_active[:sr_active*60]

f0, voiced_flag, voiced_probs = librosa.pyin(y_short, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

voiced_mask = np.zeros_like(voiced_probs, dtype=bool)
if voiced_probs is not None:
    voiced_mask = (voiced_probs > 0.6) | (voiced_flag == 1)

f0_voiced = f0[voiced_mask]
f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

def consolidate_midi_notes(midi_array, threshold_semis=0.5):
    if midi_array is None or len(midi_array) == 0:
        return np.array([])
    consolidated = []
    current_segment = [midi_array[0]]
    for note in midi_array[1:]:
        median_seg = float(np.median(current_segment))
        if abs(note - median_seg) <= threshold_semis:
            current_segment.append(note)
        else:
            consolidated.append(float(np.median(current_segment)))
            current_segment = [note]
    consolidated.append(float(np.median(current_segment)))
    return np.array(consolidated)

if len(f0_voiced) > 0:
    f0_smoothed = median_filter(f0_voiced, size=11)
    raw_midi = librosa.hz_to_midi(f0_smoothed)
    midi_notes = consolidate_midi_notes(raw_midi, threshold_semis=0.5)
else:
    midi_notes = np.array([])

# 音高差による色の生成
color_map_rgb = {
    0: (255, 0, 0), 1: (255, 69, 0), 2: (255, 140, 0), 3: (255, 215, 0),
    4: (173, 255, 47), 5: (0, 255, 0), 6: (0, 206, 209), 7: (0, 0, 255),
    8: (75, 0, 130), 9: (148, 0, 211), 10: (238, 130, 238), 11: (255, 0, 255)
}

diffs_colors_rgb = []
for i in range(len(midi_notes)-1):
    diff_exact = midi_notes[i+1] - midi_notes[i]
    diff = 0 if abs(diff_exact) < 0.5 else int(round(diff_exact)) % 12
    diffs_colors_rgb.append(color_map_rgb[diff])

# --- ここからTkinter描画部 ---
window_width, window_height = 720, 720
def rgb_to_hex(rgb): return "#%02x%02x%02x" % rgb

root = tk.Tk()
root.title("marbling melody")
root.geometry(f"{window_width}x{window_height}")

genre_color = color_map.get(final_genre, (200, 200, 200))
canvas = tk.Canvas(root, width=window_width, height=window_height, bg=rgb_to_hex(genre_color), highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

circles = []

class config:
    polygonSides = 360
    fixed_size = 30
    @staticmethod
    def racalc(): pass

class circle:
    def __init__(self, x: float, y: float, color_rgb: tuple):
        self.points = [(x, y)]
        self._points = [(config.fixed_size, 0)]
        self.position = (x, y)
        self.circle_radius = config.fixed_size
        self.side = config.polygonSides
        self.radian = ((360 / self.side) / 180) * pi
        self.color = rgb_to_hex(color_rgb)
        self.calculate_dot()
        circles.append(self)
    
    def calculate_dot(self):
        for _ in range(self.side):
            x, y = self._points[-1]
            xn = x * cos(self.radian) - y * sin(self.radian)
            yn = y * cos(self.radian) + x * sin(self.radian)
            self._points.append((xn, yn))
            self.points.append((xn + self.position[0], yn + self.position[1]))
            
    def draw(self):
        canvas.create_polygon(*self.points, fill=self.color, outline="")

# マーブリング操作用クラス
class manage_window:
    @staticmethod
    def mathematical(c, p, r):
        dx, dy = p[0] - c[0], p[1] - c[1]
        euc2 = dx**2 + dy**2
        euc = sqrt(euc2)
        distortion = sqrt(1 + (r**2 / (euc2 + 1e-6)))
        angle = 0.5 * (r / (euc + r))
        s, c_val = sin(angle), cos(angle)
        return ((dx * c_val - dy * s) * distortion + c[0], (dx * s + dy * c_val) * distortion + c[1])

    @staticmethod
    def calculate_new_circles(new_circle):
        for c_ii in circles:
            if c_ii != new_circle:
                c_ii.points = [manage_window.mathematical(new_circle.position, p, new_circle.circle_radius) for p in c_ii.points]

    @staticmethod
    def draw_circle_at_position(x, y, color_rgb):
        new_c = circle(x, y, color_rgb)
        manage_window.calculate_new_circles(new_c)

    @staticmethod
    def redraw_circle():
        canvas.delete("all")
        canvas.create_rectangle(0, 0, window_width, window_height, fill=rgb_to_hex(genre_color), outline="")
        for c in circles: c.draw()

def initialize_marbling_with_colors():
    sq_l, sq_t = window_width // 4, window_height // 4
    sq_r, sq_b = sq_l + window_width // 2, sq_t + window_height // 2
    step = max(1, len(diffs_colors_rgb) // 50)
    
    for idx in range(0, len(diffs_colors_rgb), step):
        rx = random.randint(sq_l + config.fixed_size, sq_r - config.fixed_size)
        ry = random.randint(sq_t + config.fixed_size, sq_b - config.fixed_size)
        manage_window.draw_circle_at_position(rx, ry, diffs_colors_rgb[idx])

    # 崩し処理
    if bpm > 0 and circles:
        mbc = max(1, int(len(midi_notes) / bpm)) * 10
        x_pos = [int(window_width * (i + 1) / (mbc + 1)) for i in range(mbc)]
        br_range = config.fixed_size * 3
        
        for i, x_br in enumerate(x_pos):
            shift_base = random.randint(-window_width // 5, window_width // 5)
            sorted_c = sorted(circles, key=lambda c: c.position[0], reverse=(i%2==0))
            
            for c in sorted_c:
                dist = abs(c.position[0] - x_br)
                if dist < br_range:
                    att = max(0.0, 1.0 - (dist / br_range))
                    total_s = shift_base * att * MARBLE_BREAK_INTENSITY
                    new_pts = []
                    for px, py in c.points:
                        spatial = 1.0 + 0.35 * sin((px / window_width) * 4.0 * pi + i * 0.6)
                        dx = total_s * spatial * DISPLACEMENT_SCALE
                        new_pts.append((px + dx, py))
                    c.points = new_pts

def draw_marbling_and_show_save():
    initialize_marbling_with_colors()
    manage_window.redraw_circle()
    print(f"描画完了: {final_genre}, BPM: {bpm:.2f}")
    # 保存用ボタンなどを表示
    btn = tk.Button(root, text="画像を保存", command=save_canvas)
    btn.pack(side=tk.BOTTOM, pady=10)

def save_canvas():
    f_path = filedialog.asksaveasfilename(defaultextension=".png")
    if f_path:
        root.update()
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        ImageGrab.grab(bbox=(x, y, x+canvas.winfo_width(), y+canvas.winfo_height())).save(f_path)

root.after(100, draw_marbling_and_show_save)
root.mainloop()