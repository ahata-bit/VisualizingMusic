import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    file_names = []
    genres = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            filepath = os.path.join(image_dir, filename)
            print(f"[ロード中] 画像: {filepath}")
            feature = extract_features(filepath)
            if feature is not None:
                features.append(feature)
                # ファイル名からジャンル名を抽出
                base_filename = os.path.basename(filename)  # ファイル名のみを抽出
                genre = next((g for g in ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'] if g in base_filename.lower()), 'unknown')
                file_names.append(base_filename)
                genres.append(genre)
                print(f"[ラベル付け] 画像: {filename}, ジャンル: {genre}")

    if not features:
        print("[エラー] 特徴量が抽出されませんでした。")
        return

    features = np.array(features)

    # PCAを適用
    reduced_features = apply_pca(features)

    # 階層型クラスタリングを実行
    linked = linkage(reduced_features, method='ward')

    # シルエット係数でクラスタ数を選定
    n_samples = len(reduced_features)
    max_k = min(20, n_samples - 1)
    if max_k < 2:
        print("[エラー] シルエット係数を計算するには2以上のサンプルが必要です。")
        return

    best_k = 2
    best_score = -1
    print("[シルエット] 評価開始")
    for k in range(2, max_k + 1):
        labels_k = fcluster(linked, t=k, criterion='maxclust')
        score = silhouette_score(reduced_features, labels_k)
        print(f"[シルエット] k={k}, score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = 30
    print(f"[シルエット] 最適k={best_k}, score={best_score:.4f}")

    # デンドログラムをプロット
    plt.figure(figsize=(10, 7))
    dendrogram_data = dendrogram(
        linked,
        labels=file_names,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        truncate_mode='lastp',  # クラスタ数を制限
        p=best_k  # 表示するクラスタ数
    )

    # デバッグ: デンドログラムデータを出力
    print("[デバッグ] dendrogram_data['ivl']:", dendrogram_data['ivl'])
    print("[デバッグ] dendrogram_data['leaves']:", dendrogram_data['leaves'])
    print("[デバッグ] dendrogram_data['dcoord']:", dendrogram_data['dcoord'])
    print("[デバッグ] dendrogram_data['icoord']:", dendrogram_data['icoord'])

    # 最適クラスタ数で分割
    cluster_assignments = fcluster(linked, t=best_k, criterion='maxclust')

    # クラスタ内容を収集（ファイル名とジャンル）
    cluster_contents = {}
    for idx, cluster_id in enumerate(cluster_assignments):
        cluster_contents.setdefault(cluster_id, []).append((file_names[idx], genres[idx]))

    # 各クラスタのジャンル割合を計算
    cluster_genre_ratios = {}
    for cluster_id, contents in cluster_contents.items():
        genre_counts = {}
        for _, genre in contents:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        total = sum(genre_counts.values())
        cluster_genre_ratios[cluster_id] = {genre: f"{count / total:.2%}" for genre, count in genre_counts.items()}

    # 凡例ラベルを生成
    legend_labels = []
    for cluster_id in sorted(cluster_contents.keys()):
        cluster_size = len(cluster_contents[cluster_id])
        ratio_str = ", ".join([f"{genre}: {ratio}" for genre, ratio in cluster_genre_ratios[cluster_id].items()])
        legend_labels.append(f"クラスタ {cluster_id}（{cluster_size}件）: {ratio_str}")

    # 凡例を別ウィンドウで表示
    legend_fig = plt.figure(figsize=(6, 8))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis('off')
    legend_handles = [Line2D([0], [0], color='none') for _ in legend_labels]
    legend_ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        fontsize=8,
        title="クラスタ内訳",
        title_fontsize=10,
        ncol=1
    )
    legend_fig.tight_layout()

    plt.title("画像間の階層型クラスタリング")
    plt.xlabel("画像")
    plt.ylabel("距離")
    plt.show()

if __name__ == "__main__":
    # ディレクトリを指定
    image_dir = r"E:\GC\music\classification_by_visualizationMethod_justGenre"  # 入力ディレクトリ

    generate_dendrogram(image_dir)