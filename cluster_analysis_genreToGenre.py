import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import matplotlib
from matplotlib import colors as mcolors

# 日本語フォントを設定
matplotlib.rcParams['font.family'] = 'MS Gothic'


def extract_features(image_path):
    """
    画像の特徴量を抽出する（平均色 + 色ヒストグラム）。
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_color = np.mean(image, axis=(0, 1))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    features = np.hstack([mean_color, hist])
    return features


def apply_pca(features, n_components=2):
    """PCAで次元削減して返す。"""
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced


def visualize_genre_distribution(features, labels, genres, image_paths):
    plt.figure(figsize=(16, 12), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')

    color_map = {
        'metal':     (255 / 255, 173 / 255, 173 / 255),
        'rock':      (255 / 255, 173 / 255, 214 / 255),
        'pop':       (255 / 255, 173 / 255, 255 / 255),
        'country':   (214 / 255, 173 / 255, 255 / 255),
        'disco':     (173 / 255, 173 / 255, 255 / 255),
        'blues':     (173 / 255, 214 / 255, 255 / 255),
        'jazz':      (173 / 255, 255 / 255, 255 / 255),
        'reggae':    (173 / 255, 255 / 255, 214 / 255),
        'hiphop':    (173 / 255, 255 / 255, 173 / 255),
        'classical': (214 / 255, 255 / 255, 173 / 255)
    }

    # 色の彩度を上げて白背景でも視認性を確保する（元のトーンは保つ）
    def enhance_color(rgb, sat_factor=1.6, val_factor=0.95):
        # rgb: tuple of 0-1 floats
        hsv = mcolors.rgb_to_hsv(rgb)
        h, s, v = hsv[0], hsv[1], hsv[2]
        s = min(s * sat_factor, 1.0)
        v = min(v * val_factor, 1.0)
        enhanced = mcolors.hsv_to_rgb((h, s, v))
        return tuple(float(x) for x in enhanced)

    def darken_color(rgb, dark_factor=0.35):
        # reduce value to get darker text color while keeping hue
        hsv = mcolors.rgb_to_hsv(rgb)
        h, s, v = hsv[0], hsv[1], hsv[2]
        v = max(v * dark_factor, 0.0)
        dark = mcolors.hsv_to_rgb((h, s, v))
        return tuple(float(x) for x in dark)

    enhanced_color_map = {k: enhance_color(v) for k, v in color_map.items()}

    all_x = []
    all_y = []

    x_vals = features[:, 0] if features.shape[1] >= 2 else np.array([0.0])
    y_vals = features[:, 1] if features.shape[1] >= 2 else np.array([0.0])
    x_range = float(np.ptp(x_vals)) if np.ptp(x_vals) != 0 else 1.0
    y_range = float(np.ptp(y_vals)) if np.ptp(y_vals) != 0 else 1.0
    base_offset = y_range * 0.04

    placed_labels = []  # dict list: {'x','y','half_w','half_h'}
    from matplotlib.lines import Line2D
    legend_handles = []
    legend_labels = []

    save_dir = "E:\\GC\\music\\representational_pic_v2"
    os.makedirs(save_dir, exist_ok=True)

    for genre_index, genre in enumerate(genres):
        genre_features = features[labels == genre_index]
        genre_image_paths = np.array(image_paths)[labels == genre_index]
        if genre_features.size == 0:
            continue

        cluster_center = np.mean(genre_features, axis=0)
        std_feature = np.std(genre_features, axis=0)
        # 円は中心と x 軸の標準偏差を用いる
        circle = plt.Circle(cluster_center, std_feature[0], color=enhanced_color_map.get(genre, (0.2, 0.2, 0.2)), alpha=0.8, linewidth=1.6, fill=False, linestyle='--')
        ax.add_artist(circle)

        closest_index = np.argmin(np.linalg.norm(genre_features - cluster_center, axis=1))
        representative_feature = genre_features[closest_index]
        representative_image_path = genre_image_paths[closest_index]

        # 代表点プロット
        plt.scatter(representative_feature[0], representative_feature[1], color=enhanced_color_map.get(genre, (0.2, 0.2, 0.2)), marker='x', s=260, linewidths=2)

        # ラベル配置: 点に近く、かつ既配置ラベルと重ならない位置を同心円候補から選ぶ
        px, py = representative_feature[0], representative_feature[1]
        # ラベルの推定幅・高さ（やや小さめにして近接配置を許容）
        label_half_w = max(x_range * 0.08, 1e-6)
        label_half_h = max(y_range * 0.04, 1e-6)

        def collides_with_placed_rect(cx, cy, half_w, half_h):
            for p in placed_labels:
                if abs(cx - p['x']) < (half_w + p['half_w']) and abs(cy - p['y']) < (half_h + p['half_h']):
                    return True
            return False

        # 角度分解能は維持しつつ、半径候補の最初を小さくして点に近い配置を優先
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        # 最初の半径を小さくして、上配置時にもより点に近い候補を優先
        radii = [base_offset * f for f in (0.35, 0.6, 0.9, 1.3, 1.9, 3.0)]
        chosen_x, chosen_y = px, py + base_offset

        # すべてのラベルを対応する点の真上に表示する（重なりが生じても上表示を優先）
        direct_above_x, direct_above_y = px, py + base_offset * 0.45
        chosen_x, chosen_y = direct_above_x, direct_above_y

        # 少し垂直方向の余白を追加（縮小して点にさらに近づける）
        extra_vpad = y_range * 0.008
        if chosen_y > py:
            valign = 'bottom'
            chosen_y += extra_vpad
        else:
            valign = 'top'
            chosen_y -= extra_vpad

        # ラベル描画（プロット上にはジャンル名のみ表示、代表画像情報は凡例に記載）
        # ラベルは見やすくするため暗めの色を使用（元の色相を保つ）
        base_col = enhanced_color_map.get(genre, (0.15, 0.15, 0.15))
        txt_color = darken_color(base_col, dark_factor=0.28)
        plt.text(chosen_x, chosen_y, genre, fontsize=22, color=txt_color, ha='center', va=valign)

        # 凡例用ハンドルに代表画像情報を追加（ジャンル: ファイル名）
        proxy = Line2D([0], [0], marker='x', color=enhanced_color_map.get(genre, (0.2, 0.2, 0.2)), linestyle='None', markersize=10)
        legend_handles.append(proxy)
        legend_labels.append(f"{genre}: {os.path.basename(representative_image_path)}")

        # 既配置ラベルと区別できるよう原点（点座標）も保存
        placed_labels.append({'x': chosen_x, 'y': chosen_y, 'half_w': label_half_w * 1.2, 'half_h': label_half_h * 1.2, 'origin_x': px, 'origin_y': py})

        # 代表画像を保存
        save_path = os.path.join(save_dir, os.path.basename(representative_image_path))
        try:
            cv2.imwrite(save_path, cv2.imread(representative_image_path))
        except Exception:
            pass

        all_x.extend(genre_features[:, 0])
        all_y.extend(genre_features[:, 1])

    if all_x and all_y:
        plt.xlim(min(all_x) , max(all_x)-20)
        plt.ylim(min(all_y) + 10, max(all_y)-20)

    # 収集した凡例ハンドルがあれば表示（代表画像ファイル名を記載）
    # ※一時的に凡例を非表示にするため、以下をコメントアウトしています。
    # if legend_handles:
    #     leg = plt.legend(legend_handles, legend_labels, loc='upper right', fontsize=14, frameon=True)
    #     leg.get_frame().set_alpha(0.0)
    #     for txt in leg.get_texts():
    #         txt.set_color('white')

    plt.xlabel("主成分1", fontsize=30, color="black")
    plt.ylabel("主成分2", fontsize=30, color="black")
    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.title("ジャンルごとの画像分布", fontsize=40, color="black")
    plt.tight_layout(pad=0.8)
    # 下のラベルが見切れないよう余白を追加
    plt.subplots_adjust(bottom=0.12, top=0.95)
    plt.show()


if __name__ == "__main__":
    image_dir = "E:\\GC\\music\\classification_by_visualizationMethod"
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    features = []
    labels = []
    image_paths = []

    for genre_index, genre in enumerate(genres):
        genre_path = os.path.join(image_dir, genre)
        sub_folder = f"{genre}To{genre.capitalize()}"
        sub_folder_path = os.path.join(genre_path, sub_folder)

        if not os.path.isdir(sub_folder_path):
            print(f"[警告] ジャンル '{sub_folder}' のフォルダが見つかりません。")
            continue

        for filename in os.listdir(sub_folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                filepath = os.path.join(sub_folder_path, filename)
                feature = extract_features(filepath)
                if feature is not None:
                    features.append(feature)
                    labels.append(genre_index)
                    image_paths.append(filepath)

    if not features:
        print("[エラー] 特徴量が抽出されませんでした。")
    else:
        features = np.array(features)
        labels = np.array(labels)
        reduced_features = apply_pca(features)
        visualize_genre_distribution(reduced_features, labels, genres, image_paths)