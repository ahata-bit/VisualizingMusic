import os
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2
import matplotlib

# 日本語フォントを設定
matplotlib.rcParams['font.family'] = 'MS Gothic'

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

def extract_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mean_color = np.mean(image, axis=(0, 1))
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return np.hstack([mean_color, hist])


def collect_images(root_dir):
    features = []
    file_names = []
    group_labels = []

    for group_name in sorted(os.listdir(root_dir)):
        group_path = os.path.join(root_dir, group_name)
        if not os.path.isdir(group_path):
            continue

        for filename in sorted(os.listdir(group_path)):
            if not filename.lower().endswith(IMAGE_EXTS):
                continue
            filepath = os.path.join(group_path, filename)
            print(f"[ロード中] 画像: {filepath}")
            feature = extract_features(filepath)
            if feature is None:
                continue
            features.append(feature)
            file_names.append(filename)
            group_labels.append(group_name)
            print(f"[ラベル付け] 画像: {filename}, グループ: {group_name}")

    return np.array(features), file_names, group_labels


def choose_k_silhouette(features, linked, max_k=20):
    n_samples = len(features)
    max_k = min(max_k, n_samples - 1)
    if max_k < 2:
        raise ValueError("シルエット係数を計算するには2以上のサンプルが必要です。")

    best_k = 2
    best_score = -1
    print("[シルエット] 評価開始")
    for k in range(2, max_k + 1):
        labels_k = fcluster(linked, t=k, criterion='maxclust')
        score = silhouette_score(features, labels_k)
        print(f"[シルエット] k={k}, score={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k
    print(f"[シルエット] 最適k={best_k}, score={best_score:.4f}")
    return best_k


def plot_dendrogram_with_legend(linked, file_names, group_labels, k):
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
        cluster_contents.setdefault(cluster_id, []).append(group_labels[idx])

    cluster_group_ratios = {}
    for cluster_id, contents in cluster_contents.items():
        counts = {}
        for label in contents:
            counts[label] = counts.get(label, 0) + 1
        total = sum(counts.values())
        cluster_group_ratios[cluster_id] = {label: f"{count / total:.2%}" for label, count in counts.items()}

    legend_labels = []
    for cluster_id in sorted(cluster_contents.keys()):
        cluster_size = len(cluster_contents[cluster_id])
        ratio_str = ", ".join([f"{label}: {ratio}" for label, ratio in cluster_group_ratios[cluster_id].items()])
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
    image_dir = r"E:\GC\music\output_images"
    use_scaling = False
    pca_components = 2

    features, file_names, group_labels = collect_images(image_dir)
    if len(features) == 0:
        print("[エラー] 特徴量が抽出されませんでした。")
        return

    if use_scaling:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    pca = PCA(n_components=pca_components)
    reduced_features = pca.fit_transform(features)

    linked = linkage(reduced_features, method='ward')
    best_k = choose_k_silhouette(reduced_features, linked, max_k=20)

    plot_dendrogram_with_legend(linked, file_names, group_labels, best_k)


if __name__ == "__main__":
    main()
