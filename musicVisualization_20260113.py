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
file_path = input("解析したい音声ファイルのパスを入力してください: ").strip().strip('"')

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
print(f"検出されたBPM: {bpm:.2f}")
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
    print(f"検出されたジャンル: {genre_list[predicted_label]}")
else:
    print("エラー: 推論結果がありません。音声ファイルやwindow_sizeを確認してください。")
#---------------------------------------------------------

#マーブリング---------------------------------------------------
# 最初の1分のみ読み込み
y, sr = librosa.load(file_path, sr=22050, duration=60)

# 基本ピッチ推定（pyin）
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

# 診断情報出力
num_frames = 0 if f0 is None else len(f0)
num_voiced_prob = 0 if voiced_probs is None else np.sum(~np.isnan(voiced_probs))
mean_voiced_prob = float(np.nanmean(voiced_probs)) if voiced_probs is not None else float('nan')
print(f"pyin frames: {num_frames}, voiced_probs non-nan: {num_voiced_prob}, mean voiced_probs: {mean_voiced_prob:.3f}")

# 有声音判定の閾値を下げて柔軟にする（厳しすぎると空になることがある）
voiced_mask = np.zeros_like(voiced_probs, dtype=bool)
if voiced_probs is not None:
    # 0.6 を閾値にし、pyin が返す voiced_flag も活用する
    voiced_mask = (voiced_probs > 0.6) | (voiced_flag == 1)

f0_voiced = f0[voiced_mask]
f0_voiced = f0_voiced[~np.isnan(f0_voiced)]

if len(f0_voiced) < 2:
    print("警告: pyinで十分な有声音が得られませんでした。piptrackでフォールバックします。")
    # piptrack で代替的にピッチ抽出
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    fallback_notes = []
    for i in range(pitches.shape[1]):
        col_mag = mags[:, i]
        idx = np.argmax(col_mag)
        pitch_hz = pitches[idx, i]
        if pitch_hz > 0:
            fallback_notes.append(pitch_hz)

    if len(fallback_notes) >= 2:
        f0_voiced = np.array(fallback_notes)
        print(f"piptrack で検出したピッチ数: {len(f0_voiced)}")
    else:
        print("フォールバックでもピッチが十分に検出できませんでした。マーブリング用の色が生成されません。")

# メディアンフィルタでノイズ除去（ロングトーンに対してより強めに平滑化）
def consolidate_midi_notes(midi_array, threshold_semis=0.5):
    """連続する近いピッチをまとまりとして統合して、ロングトーンの分裂を防ぐ。
    threshold_semis: 同一とみなす閾値（半音）
    戻り値は各セグメントの中央値（float配列）。"""
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
    # ロングトーンをつぶすために中央値フィルタを大きめに（11フレーム）
    f0_smoothed = median_filter(f0_voiced, size=11)
    # MIDI に変換
    raw_midi = librosa.hz_to_midi(f0_smoothed)
    # 連続する近いピッチを1つにまとめる
    midi_notes = consolidate_midi_notes(raw_midi, threshold_semis=0.5)
else:
    midi_notes = np.array([])

print(f"検出された音符数（統合後）: {len(midi_notes)}")

# オンセット分割ベースの代替抽出（ピアノ等、短い独立した音符が並ぶ場合の改善）
try:
    onsets = librosa.onset.onset_detect(y=y, sr=sr, units='samples', backtrack=False, hop_length=512)
    if len(onsets) >= 4:
        print(f"onset 検出数: {len(onsets)} -> onsetベース抽出を試行します")
        onset_samples = list(onsets) + [len(y)]
        onset_pitches = []
        for i in range(len(onset_samples)-1):
            s = onset_samples[i]
            e = onset_samples[i+1]
            seg = y[s:e]
            if len(seg) < 256:
                continue
            pitches, mags = librosa.piptrack(y=seg, sr=sr)
            frame_pitches = []
            for j in range(pitches.shape[1]):
                idx = np.argmax(mags[:, j])
                p = pitches[idx, j]
                if p > 0:
                    frame_pitches.append(p)
            if frame_pitches:
                onset_pitches.append(float(np.median(frame_pitches)))

        if len(onset_pitches) >= 2:
            onset_midi = librosa.hz_to_midi(np.array(onset_pitches))
            onset_midi_cons = consolidate_midi_notes(onset_midi, threshold_semis=0.5)
            print(f"onsetベース検出音符数（統合後）: {len(onset_midi_cons)}")
            # オンセット法の方がノート数が多ければ採用（より分解能があると判断）
            if len(onset_midi_cons) >= max(2, len(midi_notes)):
                midi_notes = onset_midi_cons
                print("onsetベースのピッチ列を採用しました。")
except Exception as e:
    print(f"onsetベース抽出でエラー: {e}")
# --- F0スムージング終了 ---

color_map_rgb = {
    0: (255, 0, 0),    # ユニゾン (赤) 
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

print(f"生成された色の数: {len(diffs_colors_rgb)}")

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
    fixed_size = 30  # サイズを少し小さくして、より多くの円を描画可能に
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

    print(f"マーブリング描画を開始します...")
    circle_count = 0
    
    # より多くの円を描画するように変更（5個に1個 → より密度を上げる）
    step = max(1, len(diffs_colors_rgb) // 50)  # 最大50個の円を描画
    
    for idx in range(0, len(diffs_colors_rgb), step):
        color_rgb = diffs_colors_rgb[idx]
        if color_rgb is None:
            continue
        
        min_x = square_left + config.fixed_size
        max_x = square_right - config.fixed_size
        min_y = square_top + config.fixed_size
        max_y = square_bottom - config.fixed_size
        
        if min_x < max_x and min_y < max_y:
            rand_x = random.randint(min_x, max_x)
            rand_y = random.randint(min_y, max_y)
            manage_window.draw_circle_at_position(rand_x, rand_y, color_rgb)
            circle_count += 1
        else:
            print("警告: ウィンドウサイズが円の半径に対して小さすぎます。中心に描画します。")
            manage_window.draw_circle_at_position(window_width / 2, window_height / 2, color_rgb)
            circle_count += 1
    
    print(f"{circle_count}個の円を描画しました")
    
    # BPMに基づくマーブリングの崩し処理
    if bpm > 0 and len(circles) > 0:
        marble_break_count = max(1, int((len(midi_notes) / bpm)*5))
        #marble_break_count = min(marble_break_count, 5)  # 最大5箇所まで
        
        # x軸方向に伝播するように、y方向に沿った崩し位置を選ぶ
        if marble_break_count == 1:
            y_positions = [window_height // 2]
        elif marble_break_count == 2:
            y_positions = [window_height // 4, window_height * 3 // 4]
        else:
            y_positions = [int(window_height * (i + 1) / (marble_break_count + 1)) for i in range(marble_break_count)]

        break_height = window_height // 5
        break_range = config.fixed_size * 3

        print(f"マーブリング崩し処理を{marble_break_count}箇所で実行します...")
        
        for y_break in y_positions:
            # 伝播用の初期シフト値（y方向のずれ）
            shift = random.randint(-break_height, break_height)
            # 伝播度合い（0.5なら前のずれの半分を次に伝える）
            propagate = 0.5

            # 円をx座標順にソート（左から右へ伝播させるイメージ）
            sorted_circles = sorted(circles, key=lambda c: c.position[0])
            prev_shift = shift

            for c in sorted_circles:
                # 円の中心が崩し位置y_break付近ならy方向にずらす
                center_y = c.position[1]
                dist = abs(center_y - y_break)
                if dist < break_range:
                    # 距離に応じて減衰させる（中心付近で最大、端で0）
                    attenuation = max(0.0, 1.0 - (dist / break_range))
                    # 前のずれに追従してずらす + 少しランダムノイズ
                    this_shift = prev_shift * propagate + random.randint(-break_height//10, break_height//10)
                    this_shift = this_shift * attenuation
                    # 非線形な歪み（サインで微妙な波打ちを追加）
                    wave = int(5 * sin((c.position[0] / window_width) * 2 * pi))
                    total_shift = int(this_shift) + wave
                    new_points = [(x, y + total_shift) for (x, y) in c.points]
                    c.points = new_points
                    # 位置も更新
                    c.position = (c.position[0], c.position[1] + total_shift)
                    prev_shift = this_shift
        
        print("マーブリング崩し処理が完了しました")

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
        # 個別の円描画はせず、最後にまとめて描画


    @staticmethod
    def redraw_circle():
        # 背景を再描画
        canvas.delete("all")
        canvas.create_rectangle(0, 0, window_width, window_height, fill=rgb_to_hex(genre_color), outline="")
        
        # すべての円を描画
        for circle_i in circles:
            circle_i.draw()
        
        # ジャンル名とBPMを描画
        # ジャンルとBPMはキャンバス上に描画しない（端末に出力する）
        pass
            
def open_save_window():
    save_win = tk.Toplevel(root)
    save_win.title("画像保存")
    save_win.geometry("200x100")
    btn = tk.Button(save_win, text="画像を保存", command=save_canvas)
    btn.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)

# --- 修正：マーブリング描画を root.after() で遅延実行 ---
def draw_marbling_and_show_save():
    print("マーブリング描画処理を開始...")
    initialize_marbling_with_colors()
    
    # 最終的な描画
    manage_window.redraw_circle()
    print("マーブリング描画が完了しました")
    # ジャンルとBPMを端末に出力（ビジュアルに表示しない）
    try:
        print(f"検出されたジャンル: {genre_list[predicted_label]}")
        print(f"検出されたBPM: {bpm:.2f}")
    except Exception as e:
        print(f"ジャンル/BPM出力エラー: {e}")
    
    # 保存ウィンドウを表示
    root.after(500, open_save_window)

# イベントループ開始後に描画を実行
root.after(100, draw_marbling_and_show_save)
root.mainloop()
