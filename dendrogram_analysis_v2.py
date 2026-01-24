import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
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

def generate_dendrogram(image_dir, genres):
    """
    デンドログラムを生成して表示する。

    Args:
        image_dir (str): 画像が保存されているディレクトリ。
        genres (list): 分析対象のジャンル名のリスト。
    """
    features = []
    labels = []

    for genre_index, genre in enumerate(genres):
        genre_path = os.path.join(image_dir, genre)
        sub_folder = f"{genre}To{genre.capitalize()}"
        sub_folder_path = os.path.join(genre_path, sub_folder)

        if not os.path.isdir(sub_folder_path):
            print(f"[警告] ジャンル '{sub_folder}' のフォルダが見つかりません。")
            continue

        genre_features = []
        for filename in os.listdir(sub_folder_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                filepath = os.path.join(sub_folder_path, filename)
                print(f"[ロード中] 画像: {filepath}")
                feature = extract_features(filepath)
                if feature is not None:
                    genre_features.append(feature)

        if genre_features:
            # 代表的な画像を選択（平均に最も近い画像）
            mean_feature = np.mean(genre_features, axis=0)
            closest_index = np.argmin(np.linalg.norm(genre_features - mean_feature, axis=1))
            representative_feature = genre_features[closest_index]
            features.append(representative_feature)
            labels.append(genre)

    if not features:
        print("[エラー] 特徴量が抽出されませんでした。")
        return

    features = np.array(features)

    # 階層型クラスタリングを実行
    linked = linkage(features, method='ward')

    # デンドログラムをプロット
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title("ジャンル間の階層型クラスタリング（代表画像）")
    plt.xlabel("ジャンル")
    plt.ylabel("距離")
    plt.show()

if __name__ == "__main__":
    # ディレクトリを指定
    image_dir = "E:\GC\music\classification_by_visualizationMethod"

    # 分析対象のジャンル
    genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

    generate_dendrogram(image_dir, genres)