# 楽曲推薦・検索の支援を目的とした音楽の特徴空間分析に基づく可視化システムの構築
# Construction of a Visualization System Based on Music Feature Space Analysis for Recommendation and Search Support

### 環境
python3.11.9

### 使用ライブラリ
librosa,
numpy,
collections,
tensorflow,
tkinter,
random,
sys,
math,
PIL,
scipy.ndimage,
os,
glob,
time

### 使用しているジャンル推定モデル
https://github.com/KMASAHIRO/music2vec

### 使用したデータセット
https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification?resource=download

・musicVisualization.py 読み込みたい楽曲パスを入力して、楽曲の視覚化を行う.

・cluster_analysis_output_images.py データセット内の全楽曲から出力された画像を検証.

・cluster_analysis_genreToGenre.py  生成された画像の内からジャンル推定のがさをはじいた画像を検証.

### 生成された画像
https://drive.google.com/drive/folders/1fuFEa3dFwIw9B6aDpteEFwiVyoz-2xY-?usp=drive_link


