import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import matplotlib

# 日本語フォントを設定
matplotlib.rcParams['font.family'] = 'MS Gothic'  # Windows環境で日本語対応のフォントを指定

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


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


def visualize_group_distribution(features, labels, groups, image_paths):
    """
    各グループの代表的な画像の点のみプロットし、その他の画像が存在する範囲を円で囲む。

    Args:
        features (np.array): 抽出された特徴量。
        labels (np.array): グループラベル。
        groups (list): 分析対象のグループ名のリスト。
        image_paths (list): 各特徴量に対応する画像のパス。
    """
    plt.figure(figsize=(12, 8))
    plt.gca().set_facecolor("black")

    color_map = {}
    cmap = plt.cm.get_cmap('tab10', len(groups))
    for i, group in enumerate(groups):
        color_map[group] = cmap(i)

    all_x = []
    all_y = []

    save_dir = "E:\\GC\\music\\representational_pic_v2"
    os.makedirs(save_dir, exist_ok=True)

    for group_index, group in enumerate(groups):
        group_features = features[labels == group_index]
        group_image_paths = np.array(image_paths)[labels == group_index]

        if len(group_features) == 0:
            continue

        cluster_center = np.mean(group_features, axis=0)
        std_feature = np.std(group_features, axis=0)
        circle = plt.Circle(
            cluster_center,
            std_feature[0],
            color=color_map.get(group, (0.5, 0.5, 0.5)),
            alpha=0.5,
            fill=False,
            linestyle='--',
            label=f"範囲: {group}"
        )
        plt.gca().add_artist(circle)

        closest_index = np.argmin(np.linalg.norm(group_features - cluster_center, axis=1))
        representative_feature = group_features[closest_index]
        representative_image_path = group_image_paths[closest_index]
        plt.scatter(
            representative_feature[0],
            representative_feature[1],
            color=color_map.get(group, (0.5, 0.5, 0.5)),
            marker='x',
            s=100,
            label=f"代表画像: {os.path.basename(representative_image_path)}"
        )

        plt.text(
            representative_feature[0],
            representative_feature[1] + 5,
            os.path.basename(representative_image_path),
            fontsize=8,
            color=color_map.get(group, (0.5, 0.5, 0.5))
        )

        save_path = os.path.join(save_dir, os.path.basename(representative_image_path))
        cv2.imwrite(save_path, cv2.imread(representative_image_path))

        all_x.extend(group_features[:, 0])
        all_y.extend(group_features[:, 1])

    if all_x and all_y:
        plt.xlim(min(all_x) - 10, max(all_x))
        plt.ylim(min(all_y) - 10, max(all_y))

    plt.xlabel("主成分1", fontsize=12, color="black")
    plt.ylabel("主成分2", fontsize=12, color="black")

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label.startswith("代表画像"):
            new_labels.append(label.replace("代表画像: ", "画像ファイル: "))
        else:
            new_labels.append(label)
    plt.legend(handles, new_labels, bbox_to_anchor=(1.01, 0.5), loc='center left', fontsize=10, title="凡例", title_fontsize=12)

    plt.title("グループごとの画像分布", fontsize=12, color="black")
    plt.tight_layout(pad=0.5)
    plt.show()


def main():
    image_dir = "E:\\GC\\music\\output_images"

    features = []
    labels = []
    image_paths = []
    groups = []

    group_names = [d for d in sorted(os.listdir(image_dir)) if os.path.isdir(os.path.join(image_dir, d))]
    for group_index, group in enumerate(group_names):
        group_path = os.path.join(image_dir, group)
        groups.append(group)

        for filename in sorted(os.listdir(group_path)):
            if filename.lower().endswith(IMAGE_EXTS):
                filepath = os.path.join(group_path, filename)
                feature = extract_features(filepath)
                if feature is not None:
                    features.append(feature)
                    labels.append(group_index)
                    image_paths.append(filepath)

    if not features:
        print("[エラー] 特徴量が抽出されませんでした。")
        return

    features = np.array(features)
    labels = np.array(labels)

    reduced_features = apply_pca(features)
    visualize_group_distribution(reduced_features, labels, groups, image_paths)


if __name__ == "__main__":
    main()
