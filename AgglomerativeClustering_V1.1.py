import os
import numpy as np
from PIL import Image
import glob
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
from scipy.cluster.hierarchy import dendrogram, linkage

def extract_color_histogram(image_path, bins=32):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    hist_r = np.histogram(img_np[:,:,0], bins=bins, range=(0,255))[0]
    hist_g = np.histogram(img_np[:,:,1], bins=bins, range=(0,255))[0]
    hist_b = np.histogram(img_np[:,:,2], bins=bins, range=(0,255))[0]
    hist = np.concatenate([hist_r, hist_g, hist_b])
    hist = hist / np.sum(hist)  # 正規化
    return hist

def evaluate_image_folder(folder_path):
    image_files = glob.glob(os.path.join(folder_path, "*.png"))
    features = []
    names = []
    for img_path in image_files:
        hist = extract_color_histogram(img_path)
        features.append(hist)
        names.append(os.path.splitext(os.path.basename(img_path))[0])
    features = np.array(features)
    
    # 主成分分析で2次元に可視化
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(features)
    
    # --- PCごとの説明分散比を表示して軸ラベルに反映 ---
    explained = pca.explained_variance_ratio_
    pc1_pct = explained[0] * 100
    pc2_pct = explained[1] * 100
    print(f"PC1 explains {pc1_pct:.2f}% of variance, PC2 explains {pc2_pct:.2f}%")
    
    # --- KMeans クラスタリング（PC空間上で） ---
    scaler = StandardScaler()
    reduced_scaled = scaler.fit_transform(reduced)
    
    # --- 階層クラスタリング（PC空間上で） ---
    # クラスタ数を試して最適なものを探索
    best_score = -1.0
    best_n_clusters = 2
    best_linkage = 'ward'
    
    for linkage_method in ['ward', 'complete', 'average', 'single']:
        for n_clusters in range(2, min(8, len(features))):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method).fit(reduced_scaled)
            labels_temp = clusterer.labels_
            
            if len(set(labels_temp)) < 2:
                continue
            
            score = silhouette_score(reduced_scaled, labels_temp)
            print(f"linkage={linkage_method} n_clusters={n_clusters} silhouette={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
                best_linkage = linkage_method
    
    print(f"\n最適パラメータ: linkage={best_linkage}, n_clusters={best_n_clusters}, silhouette={best_score:.4f}")
    clusterer = AgglomerativeClustering(n_clusters=best_n_clusters, linkage=best_linkage).fit(reduced_scaled)
    labels = clusterer.labels_
    best_k = len(set(labels))
    print(f"クラスタ数: {best_k}")
    # --- クラスタリング終了 ---
    
    # --- デンドログラム表示 ---
    # reduced_scaled を使って階層的クラスタの linkage を計算しプロット
    Z = linkage(reduced_scaled, method=best_linkage)
    plt.figure(figsize=(12, 6))
    # 葉ラベルのフォントサイズを調整（サンプル数が多い場合は少し抑える）
    if len(names) > 50:
        leaf_fs = 12
        dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=leaf_fs,
                   truncate_mode='lastp', p=30, show_leaf_counts=True)
    else:
        leaf_fs = 14
        dendrogram(Z, labels=names, leaf_rotation=90, leaf_font_size=leaf_fs, color_threshold=None)

    # タイトル・軸ラベルのフォントサイズも大きくする
    plt.title(f"Dendrogram (linkage={best_linkage}, n={len(names)})", fontsize=18)
    plt.xlabel("Sample", fontsize=14)
    plt.ylabel("Distance", fontsize=14)
    # x/y の目盛りサイズ調整（葉ラベルにも適用）
    plt.xticks(fontsize=leaf_fs)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.show()
    # --- デンドログラムここまで ---
    
    plt.figure(figsize=(20, 8))

    # マーカーのリスト
    markers = ['o', 's', '^', 'x', 'd', '*', '+', 'p', '<', '>']  # 異なるマーカーの種類
    cmap = plt.get_cmap("tab10")
    
    for i, name in enumerate(names):
        marker = markers[i % len(markers)]  # マーカーを順番に割り当てる
        # 階層クラスタは全てのクラスタが有効なため、-1チェックなし
        color = cmap(labels[i] % 10)
        plt.scatter(reduced[i,0], reduced[i,1], label=name, s=400, marker=marker,
                   color=color, edgecolors='black', linewidth=1.5)
         
    plt.title("PCA Visualization of Image Features (Color Histogram) - Hierarchical Clustering", fontsize=38)
    plt.xlabel(f"PC1 ({pc1_pct:.2f}% var.)", fontsize=30)
    plt.ylabel(f"PC2 ({pc2_pct:.2f}% var.)", fontsize=30)
    plt.legend(fontsize=20, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- 楕円でクラスタを囲む（階層クラスタリング用） ---
    plt.figure(figsize=(20, 8))
    for i, name in enumerate(names):
         marker = markers[i % len(markers)]
         color = cmap(labels[i] % 10)
         plt.scatter(reduced[i,0], reduced[i,1], label=name, s=400, marker=marker,
                    color=color, edgecolors='black', linewidth=1.5)
     
      # 各クラスタを楕円で囲う
    for c in range(best_k):  # 階層クラスタはノイズなし
           idx_c = np.where(labels == c)[0]
           if len(idx_c) < 2:
              continue
           pts = reduced[idx_c]
           mean = pts.mean(axis=0)
           cov = np.cov(pts.T)
           # 固有値・固有ベクトルから楕円パラメータ計算
           eigvals, eigvecs = np.linalg.eigh(cov)
           angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
           width, height = 6 * np.sqrt(eigvals)  # 3標準偏差に拡張（より広い範囲）
           ellipse_color = cmap(c % 10)
           ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                           facecolor=ellipse_color, alpha=0.05, edgecolor=ellipse_color, linewidth=3)
           plt.gca().add_patch(ellipse)
     
    plt.title("PCA Visualization with Cluster Ellipses (Color Histogram) - Hierarchical Clustering", fontsize=38)
    plt.xlabel(f"PC1 ({pc1_pct:.2f}% var.)", fontsize=30)
    plt.ylabel(f"PC2 ({pc2_pct:.2f}% var.)", fontsize=30)
    plt.legend(fontsize=18, bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0.)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
     
     # --- クラスタごとに個別プロット（重なりを避けるため） ---
    unique_labels = sorted(set(labels))
    for c in unique_labels:
         idx_c = np.where(labels == c)[0]
         plt.figure(figsize=(12, 8))
         
         # 全点をグレーで背景表示
         for i in range(len(names)):
             marker = markers[i % len(markers)]
             if labels[i] == c:
                color = cmap(c % 10)
                plt.scatter(reduced[i,0], reduced[i,1], label=names[i], s=400, marker=marker,
                            color=color, edgecolors='black', linewidth=1.5)
             else:
                 plt.scatter(reduced[i,0], reduced[i,1], s=100, marker='o',
                            color='lightgray', alpha=0.3)
         
         # クラスタの楕円描画
         if len(idx_c) >= 2:
             pts = reduced[idx_c]
             mean = pts.mean(axis=0)
             cov = np.cov(pts.T)
             eigvals, eigvecs = np.linalg.eigh(cov)
             angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
             width, height = 6 * np.sqrt(eigvals)
             ellipse_color = cmap(c % 10)
             ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                             facecolor=ellipse_color, alpha=0.1, edgecolor=ellipse_color, linewidth=3)
             plt.gca().add_patch(ellipse)
         
    cluster_label = f"Cluster {c}"
    plt.title(f"{cluster_label} (n={len(idx_c)} images)", fontsize=20)
    plt.xlabel(f"PC1 ({pc1_pct:.2f}% var.)", fontsize=16)
    plt.ylabel(f"PC2 ({pc2_pct:.2f}% var.)", fontsize=16)
    plt.legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
     # --- 個別プロット終了 ---
    
    # 類似度行列も計算可能
    similarity = np.dot(features, features.T)
    print("Similarity Matrix Between Images:")
    print(similarity)
    
    # 類似度行列のヒートマップ表示
    plt.figure(figsize=(8,6))
    plt.imshow(similarity, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Similarity')
    plt.xticks(ticks=np.arange(len(names)), labels=names, rotation=90, fontsize=10)
    plt.yticks(ticks=np.arange(len(names)), labels=names, fontsize=10)
    plt.title("Similarity Matrix Between Images (Heatmap)", fontsize=14)
    plt.tight_layout()
    plt.show()

folder_path = input("入力してください")
evaluate_image_folder(folder_path)