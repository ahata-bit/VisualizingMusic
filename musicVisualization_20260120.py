import librosa
import numpy as np
from collections import Counter
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog
import random
from math import cos, sin, sqrt, pi, atan2
from PIL import Image, ImageGrab
from scipy.ndimage import median_filter
import os
import glob
import time

# ==========================================
# 設定値
# ==========================================
MARBLE_BREAK_WEIGHT = 1.0
MARBLE_BREAK_INTENSITY = 2.0
DISPLACEMENT_SCALE = 1.0
SAVE_DIR = "E:/GC/music/output_images/"  # 画像の保存先
os.makedirs(SAVE_DIR, exist_ok=True)

# ジャンル定義
MODEL_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
VISUAL_GENRE_ORDER = ['metal','rock','pop','country','disco','blues','jazz','reggae','hiphop','classical']

# ジャンルの感性的なグラデーション順に色を定義する
# ジャンル名: (R, G, B)
color_map = {
    'metal':     (255, 173, 173), # 赤系
    'rock':      (255, 173, 214),
    'pop':       (255, 173, 255),
    'country':   (214, 173, 255),
    'disco':     (173, 173, 255), # 青系
    'blues':     (173, 214, 255),
    'jazz':      (173, 255, 255),
    'reggae':    (173, 255, 214),
    'hiphop':    (173, 255, 173),
    'classical': (214, 255, 173)  # 緑系
}

# モデルのインデックス定義（これはモデルの仕様なので変えない）
MODEL_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
color_map_rgb_pitch = {
    0: (255, 0, 0), 1: (255, 69, 0), 2: (255, 140, 0), 3: (255, 215, 0),
    4: (173, 255, 47), 5: (0, 255, 0), 6: (0, 206, 209), 7: (0, 0, 255),
    8: (75, 0, 130), 9: (148, 0, 211), 10: (238, 130, 238), 11: (255, 0, 255)
}

# モデルのロード
print("モデルをロード中...")
music2vec = tf.keras.models.load_model('E:/zemi/GC/music2vec_10epochs.keras')

# 全ファイルパスの取得 testデータを自動的に処理
folder_names = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
all_wav_files = []
for n in range(10):
    path = os.path.normpath(f'E:/GC/music/Data/genres_original/{n}/*.wav')
    all_wav_files.extend(glob.glob(path))

print(f"合計 {len(all_wav_files)} ファイルを検出しました。全自動処理を開始します。")
'''
file_path = input("解析したい音声ファイルのパスを入力してください: ").strip().strip('"')

# 音声ファイルを読み込み
y_1, sr_1 = librosa.load(file_path, sr=22050)
all_wav_files = [file_path]
'''
# ==========================================
# クラス定義 (元のロジックを完全保持)
# ==========================================
class config:
    polygonSides = 360
    fixed_size = 30
    @staticmethod
    def racalc(): pass

class circle:
    def __init__(self, x, y, color_rgb):
        self.points = [(x, y)]
        self._points = []
        self.position = (x, y)
        self.circle_radius = config.fixed_size
        self._points.append((self.circle_radius, 0))
        self.side = config.polygonSides
        self.angel = 360 / self.side
        self.radian = (self.angel / 180) * pi
        self.color = "#%02x%02x%02x" % color_rgb
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
        # 隙間を埋めるためにアウトラインの幅を調整
        canvas.create_polygon(*self.points, fill=self.color, outline=self.color, width=2)

class manage_window:
    @staticmethod
    def mathematical(c, p, r):
        dx, dy = p[0] - c[0], p[1] - c[1]
        euc2 = dx**2 + dy**2
        euc = sqrt(euc2)
        distortion = sqrt(1 + (r**2 / (euc2 + 1e-6)))
        strength = 0.5
        angle = strength * (r / (euc + r))
        s, cv = sin(angle), cos(angle)
        rx = (dx * cv - dy * s) * distortion
        ry = (dx * s + dy * cv) * distortion
        return (rx + c[0], ry + c[1])

    @staticmethod
    def calculate_new_circles(new_circle):
        for circ in circles:
            if circ != new_circle:
                circ.points = [manage_window.mathematical(new_circle.position, p, new_circle.circle_radius) for p in circ.points]

    @staticmethod
    def apply_fluid_drag(circle_obj, direction='horizontal'):
        phase = random.uniform(0, 2 * pi)
        for i in range(len(circle_obj.points)):
            px, py = circle_obj.points[i]
            if direction == 'horizontal':
                circle_obj.points[i] = (px + 15 * sin(py / 50.0 + phase), py)
            else:
                circle_obj.points[i] = (px, py + 15 * sin(px / 50.0 + phase))

def consolidate_midi_notes(midi_array, threshold_semis=0.5):
    if midi_array is None or len(midi_array) == 0: return np.array([])
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

# ==========================================
# Tkinter セットアップ
# ==========================================
window_width, window_height = 720, 720
root = tk.Tk()
root.title("Auto Marbling Melody")
canvas = tk.Canvas(root, width=window_width, height=window_height, highlightthickness=0)
canvas.pack()

# ==========================================
# メインループ関数
# ==========================================
file_idx = 0
circles = []

def run_auto_process():
    global file_idx, circles
    if file_idx >= len(all_wav_files):
        print("完了しました。")
        root.destroy()
        return

    file_path = all_wav_files[file_idx]
    filename = os.path.basename(file_path)
    print(f"\n--- [{file_idx+1}/{len(all_wav_files)}] {filename} ---")

    try:
        # 1. 音声読み込み
        y, sr = librosa.load(file_path, sr=22050)

        # 2. テンポ推定
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = tempo[0] if isinstance(tempo, (list, np.ndarray)) else tempo

        # 3. ジャンル推定
        # 修正: ジャンル推定ロジックをv4に近づける
        # 修正: ウィンドウサイズを小さくして細かい区間ごとにジャンル推定
        window_size = 675808  # モデルが期待する固定サイズ
        step = window_size // 2  # スライド幅を調整
        results = []

        for start in range(0, len(y) - window_size + 1, step):
            end = start + window_size
            chunk = y[start:end]
            chunk = chunk * 256.0
            chunk = np.reshape(chunk, (-1, 1))
            input_data = np.expand_dims(chunk, axis=0)  # (1, window_size, 1)
            pred = music2vec.predict(input_data)
            pred_idx = int(np.argmax(pred))
            results.append(pred_idx)

            # セグメント単位の結果を出力
            start_sec = start / sr
            end_sec = end / sr
            pred_genre = MODEL_GENRES[pred_idx]
            print(f"Segment {start_sec:.2f}s-{end_sec:.2f}s -> {pred_genre}")

        # 余り区間の処理
        remainder = len(y) % window_size
        if remainder != 0:
            chunk = y[-remainder:]
            chunk = np.pad(chunk, (0, window_size - remainder))  # ゼロパディングで長さを揃える
            chunk = chunk * 256.0
            chunk = np.reshape(chunk, (-1, 1))
            input_data = np.expand_dims(chunk, axis=0)
            pred = music2vec.predict(input_data)
            pred_idx = int(np.argmax(pred))
            results.append(pred_idx)

            # 余り区間の結果を出力
            start_sec = (len(y) - remainder) / sr
            end_sec = len(y) / sr
            pred_genre = MODEL_GENRES[pred_idx]
            print(f"Remainder {start_sec:.2f}s-{end_sec:.2f}s -> {pred_genre}")

        # 最頻値で最終ジャンルを決定
        final_genre_idx = Counter(results).most_common(1)[0][0] if results else 0
        final_genre_name = MODEL_GENRES[final_genre_idx]
        genre_color = color_map.get(final_genre_name, (200, 200, 200))

        # 4. ピッチ解析 (pyin + consolidate)
        y_short = y[:sr*60]
        f0, voiced_flag, voiced_probs = librosa.pyin(y_short, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        voiced_mask = (voiced_probs > 0.6) | (voiced_flag == 1) if voiced_probs is not None else []
        f0_voiced = f0[voiced_mask] if len(voiced_mask) > 0 else []
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)] if len(f0_voiced) > 0 else []

        if len(f0_voiced) > 0:
            midi_notes = consolidate_midi_notes(librosa.hz_to_midi(median_filter(f0_voiced, size=11)))
        else:
            midi_notes = []

        # 5. 色変換
        diffs_colors = []
        for i in range(len(midi_notes)-1):
            diff = int(round(midi_notes[i+1] - midi_notes[i])) % 12
            diffs_colors.append(color_map_rgb_pitch[diff])

        # 6. 描画
        canvas.delete("all")
        circles = []
        canvas.config(bg="#%02x%02x%02x" % genre_color)

        # 形状生成
        step = max(1, len(diffs_colors) // 50)
        sq_l, sq_t = window_width // 4, window_height // 4
        sq_r, sq_b = sq_l + window_width // 2, sq_t + window_height // 2

        for idx in range(0, len(diffs_colors), step):
            rx = random.randint(sq_l + config.fixed_size, sq_r - config.fixed_size)
            ry = random.randint(sq_t + config.fixed_size, sq_b - config.fixed_size)
            new_c = circle(rx, ry, diffs_colors[idx])
            manage_window.calculate_new_circles(new_c)
            if random.random() > 0.5:
                manage_window.apply_fluid_drag(new_c, random.choice(['horizontal', 'vertical']))

        # BPM崩し
        if bpm > 0 and circles:
            mb_count = max(1, int(len(midi_notes) / bpm)) * 10
            x_positions = [int(window_width * (i + 1) / (mb_count + 1)) for i in range(mb_count)]
            for i, x_break in enumerate(x_positions):
                shift = random.randint(-window_width // 5, window_width // 5)
                direction = 'rtl' if i % 2 == 0 else 'ltr'
                sorted_circles = sorted(circles, key=lambda c: c.position[0], reverse=(direction == 'rtl'))
                prev_shift = shift
                for c in sorted_circles:
                    dist = abs(c.position[0] - x_break)
                    if dist < config.fixed_size * 3:
                        att = max(0.0, 1.0 - (dist / (config.fixed_size * 3)))
                        this_shift = (prev_shift * 0.5 + random.randint(-7, 7)) * att * MARBLE_BREAK_INTENSITY
                        wave = 5.0 * MARBLE_BREAK_INTENSITY * sin((c.position[0] / window_width) * 2 * pi)
                        total_s = this_shift + wave
                        
                        new_pts = []
                        cx, cy = c.position
                        for px, py in c.points:
                            rel_x, rel_y = px - cx, py - cy
                            g_factor = px / window_width if direction == 'ltr' else 1.0 - (px / window_width)
                            spatial = 1.0 + 0.35 * sin((px / window_width) * 4.0 * pi + i * 0.6)
                            dx = (total_s / 144) * att * g_factor * spatial * (1.0 + 0.25 * MARBLE_BREAK_INTENSITY) * DISPLACEMENT_SCALE
                            dy = 0.12 * (total_s / 144) * cos((py / window_height) * 3.0 * pi + i * 0.6) * DISPLACEMENT_SCALE
                            
                            ang = atan2(rel_y, rel_x)
                            r_len = sqrt(rel_x**2 + rel_y**2)
                            new_ang = ang + (0.02 * att * MARBLE_BREAK_INTENSITY * DISPLACEMENT_SCALE * sin((px / window_width) * 2.0 * pi + i * 0.6))
                            new_pts.append((cx + cos(new_ang) * r_len + dx, cy + sin(new_ang) * r_len + dy))
                        c.points = new_pts
                        prev_shift = this_shift

        # 描画反映
        for c in circles: c.draw()
        root.update()

        # 7. 画像保存
        # 修正: ファイル名をWAVファイルのジャンルに基づいて決定
        # ファイルパスからジャンルを抽出
        wav_genre = "unknown"
        for genre in MODEL_GENRES:
            if genre in file_path:
                wav_genre = genre
                break

        # 修正: ファイル名にジャンル名が重複しないようにする
        # ファイル名から既存のジャンル名を削除
        if wav_genre in filename:
            cleaned_filename = filename.replace(wav_genre, '').strip('_')
        else:
            cleaned_filename = filename

        # 修正後のファイル名生成
        save_name = f"{wav_genre}_{cleaned_filename}.png"
        save_path = os.path.join(SAVE_DIR, save_name)
        # Canvas領域の座標取得
        root.update_idletasks()
        x = root.winfo_rootx() + canvas.winfo_x()
        y = root.winfo_rooty() + canvas.winfo_y()
        ImageGrab.grab(bbox=(x, y, x + window_width, y + window_height)).save(save_path)
        print(f"Success: {save_name} を保存しました。")

    except Exception as e:
        print(f"Error at {filename}: {e}")

    file_idx += 1
    root.after(100, run_auto_process)

# 実行
root.after(100, run_auto_process)
root.mainloop()