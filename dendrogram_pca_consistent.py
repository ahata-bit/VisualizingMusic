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

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_color = np.mean(image, axis=(0, 1))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return np.hstack([mean_color, hist])


def collect_data(image_dir):
    features = []
    file_names = []
    genres = []

    for genre_index, genre in enumerate(GENRES):
        genre_path = os.path.join(image_dir, genre)
        sub_folder = f"{genre}To{genre.capitalize()}"
        sub_folder_path = os.path.join(genre_path, sub_folder)

        if not os.path.isdir(sub_folder_path):
            print(f"[警告] ジャンル '{sub_folder}' のフォルダが見つかりません。")
            continue

        for filename in sorted(os.listdir(sub_folder_path)):
            if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                filepath = os.path.join(sub_folder_path, filename)
                print(f"[ロード中] 画像: {filepath}")
                feature = extract_features(filepath)
                if feature is None:
                    continue
                base_filename = os.path.basename(filename)
                features.append(feature)
                file_names.append(base_filename)
                genres.append(genre)
                print(f"[ラベル付け] 画像: {base_filename}, ジャンル: {genre}")

    return np.array(features), file_names, genres


def choose_k_silhouette(reduced_features, linked, max_k=20):
    n_samples = len(reduced_features)
    max_k = min(max_k, n_samples - 1)
    if max_k < 2:
        raise ValueError("シルエット係数を計算するには2以上のサンプルが必要です。")

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
    return best_k


def plot_dendrogram_with_legend(linked, file_names, genres, k):
    plt.figure(figsize=(10, 7))
    dendrogram(
        linked,
        labels=file_names,
        orientation='top',
        distance_sort='descending',
        show_leaf_counts=True,
        truncate_mode='lastp',
        p=k
    )

    cluster_assignments = fcluster(linked, t=k, criterion='maxclust')
    cluster_contents = {}
    for idx, cluster_id in enumerate(cluster_assignments):
        cluster_contents.setdefault(cluster_id, []).append((file_names[idx], genres[idx]))

    cluster_genre_ratios = {}
    for cluster_id, contents in cluster_contents.items():
        genre_counts = {}
        for _, genre in contents:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        total = sum(genre_counts.values())
        cluster_genre_ratios[cluster_id] = {genre: f"{count / total:.2%}" for genre, count in genre_counts.items()}

    legend_labels = []
    for cluster_id in sorted(cluster_contents.keys()):
        cluster_size = len(cluster_contents[cluster_id])
        ratio_str = ", ".join([f"{genre}: {ratio}" for genre, ratio in cluster_genre_ratios[cluster_id].items()])
        legend_labels.append(f"クラスタ {cluster_id}（{cluster_size}件）: {ratio_str}")

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


def main():
    image_dir = r"E:\GC\music\classification_by_visualizationMethod"
    features, file_names, genres = collect_data(image_dir)

    if len(features) == 0:
        print("[エラー] 特徴量が抽出されませんでした。")
        return

    # PCA（cluster_analysisと同じく標準化なし）
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(features)

    # 階層型クラスタリング（PCA後）
    linked = linkage(reduced_features, method='ward')

    # シルエット係数でk選定
    best_k = choose_k_silhouette(reduced_features, linked, max_k=20)

    # デンドログラムと凡例
    plot_dendrogram_with_legend(linked, file_names, genres, best_k)


if __name__ == "__main__":
    main()
