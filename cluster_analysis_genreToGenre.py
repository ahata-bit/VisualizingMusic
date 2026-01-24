import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import matplotlib

# 日本語フォントを設定
matplotlib.rcParams['font.family'] = 'MS Gothic'  # Windows環境で日本語対応のフォントを指定

def extract_features(image_path):
    """
    画像の特徴量を抽出する。
    現在は平均色と色ヒストグラムを特徴量として使用。
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # BGRからRGBに変換
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 平均色を計算
    mean_color = np.mean(image, axis=(0, 1))

    # 色ヒストグラムを計算
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    # 平均色とヒストグラムを結合
    features = np.hstack([mean_color, hist])
    return features

def apply_pca(features, n_components=2):
    """
    特徴量の次元削減を行う。

    Args:
        features (np.array): 特徴量データ。
        n_components (int): 次元数。

    Returns:
        np.array: 次元削減後の特徴量。
    """
    pca = PCA(n_components=n_components)
    reduced_features = pca.fit_transform(features)
    return reduced_features

def visualize_genre_distribution(features, labels, genres, image_paths):
    """
    各ジャンルの代表的な画像の点のみプロットし、その他の画像が存在する範囲を円で囲む。

    Args:
        features (np.array): 抽出された特徴量。
        labels (np.array): クラスターラベル。
        genres (list): 分析対象のジャンル名のリスト。
        image_paths (list): 各特徴量に対応する画像のパス。
    """
    plt.figure(figsize=(12, 8))  # 図のサイズをコンパクトに調整

    # 背景色を黒に設定
    plt.gca().set_facecolor("black")

    # カラーマップを定義
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

    all_x = []
    all_y = []

    # 保存先ディレクトリ
    save_dir = "E:\\GC\\music\\representational_pic_v2"
    os.makedirs(save_dir, exist_ok=True)

    for genre_index, genre in enumerate(genres):
        # ジャンルごとの特徴量と画像パスを抽出
        genre_features = features[labels == genre_index]
        genre_image_paths = np.array(image_paths)[labels == genre_index]

        # クラスター中心を計算
        cluster_center = np.mean(genre_features, axis=0)

        # 円を描画（中心: クラスター中心、半径: 標準偏差）
        std_feature = np.std(genre_features, axis=0)
        circle = plt.Circle(cluster_center, std_feature[0], color=color_map.get(genre, (0.5, 0.5, 0.5)), alpha=0.5, fill=False, linestyle='--', label=f"範囲: {genre}")
        plt.gca().add_artist(circle)

        # 代表画像を選定
        closest_index = np.argmin(np.linalg.norm(genre_features - cluster_center, axis=1))
        representative_feature = genre_features[closest_index]
        representative_image_path = genre_image_paths[closest_index]
        plt.scatter(representative_feature[0], representative_feature[1], color=color_map.get(genre, (0.5, 0.5, 0.5)), marker='x', s=100, label=f"代表画像: {os.path.basename(representative_image_path)}")

        # 画像名を表示
        plt.text(representative_feature[0], representative_feature[1] + 5, os.path.basename(representative_image_path), fontsize=8, color=color_map.get(genre, (0.5, 0.5, 0.5)))

        # 代表画像を保存（元のファイル名を使用）
        save_path = os.path.join(save_dir, os.path.basename(representative_image_path))
        cv2.imwrite(save_path, cv2.imread(representative_image_path))

        # 全ての座標を収集
        all_x.extend(genre_features[:, 0])
        all_y.extend(genre_features[:, 1])

    # 軸の範囲を設定
    plt.xlim(min(all_x) - 10, max(all_x))  # x軸の幅を拡大
    plt.ylim(min(all_y) - 10, max(all_y))  # y軸の幅を拡大

    # 軸ラベルを追加
    plt.xlabel("主成分1", fontsize=12, color="black")
    plt.ylabel("主成分2", fontsize=12, color="black")

    # 凡例を改善
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith("代表画像"):
            new_labels.append(label.replace("代表画像: ", "画像ファイル: "))
        else:
            new_labels.append(label)
    plt.legend(handles, new_labels, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=10, title="凡例", title_fontsize=12)

    plt.title("ジャンルごとの画像分布", fontsize=12, color="black")  # タイトルのフォントサイズを調整
    plt.tight_layout(pad=0.5)  # レイアウトの余白をさらに減らす
    plt.show()

if __name__ == "__main__":
    # ディレクトリを指定
    image_dir = "E:\\GC\\music\\classification_by_visualizationMethod"

    # 分析対象のジャンル
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

        # PCAを適用
        reduced_features = apply_pca(features)

        visualize_genre_distribution(reduced_features, labels, genres, image_paths)