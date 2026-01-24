import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
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

def generate_dendrogram(image_dir):
    """
    デンドログラムを生成して表示する。

    Args:
        image_dir (str): 画像が保存されているディレクトリ。
    """
    features = []
    labels = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            filepath = os.path.join(image_dir, filename)
            print(f"[ロード中] 画像: {filepath}")
            feature = extract_features(filepath)
            if feature is not None:
                features.append(feature)
                # ファイル名からジャンル名を抽出してラベルに設定
                genre = next((g for g in ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'] if g in filename.lower()), 'unknown')
                labels.append(genre)  # ジャンル名をラベルとして使用

    if not features:
        print("[エラー] 特徴量が抽出されませんでした。")
        return

    features = np.array(features)

    # PCAを適用
    reduced_features = apply_pca(features)

    # 階層型クラスタリングを実行
    linked = linkage(reduced_features, method='ward')

    # デンドログラムをプロット
    plt.figure(figsize=(10, 7))
    dendrogram(linked, labels=labels, orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title("画像間の階層型クラスタリング")
    plt.xlabel("画像")
    plt.ylabel("距離")
    plt.show()

if __name__ == "__main__":
    # ディレクトリを指定
    image_dir = r"E:\GC\music\representational_pic_v2"  # 入力ディレクトリ

    generate_dendrogram(image_dir)