# raman-pca
ラマンデータにおける主成分分析の結果（データ全体に対する各主成分軸の寄与率、ある軸における各変数の重み、散布図）を手軽に確認できるツール

## 導入
```bash
poetry install
```

## 使い方
### 動作確認
データが正しければエラーなく動作する
```bash
poetry run python main.py run data/hoge.csv
```

### 共通オプション
- std: 解析前の標準化
- ep: 指定した行（サンプル）をモデルの学習対象から除く
- et: 指定した行（サンプル）をPCAの変換対象から除く
```bash
poetry run python main.py -ep row1 -ep row2 run data/hoge.csv
```

### データ全体に対する各主成分軸の寄与率
```bash
# 1% 以上の寄与率を持つ主成分軸を出力
poetry run python main.py explained data/hoge.csv
```

### 任意の主成分軸における各変数の重み
- ax: 主成分軸
- n: 上位n個までの変数とその重みを出力
```bash
# PC1 における重みが高い変数を上位20個まで出力する
poetry run python main.py weights data/hoge.csv -ax 1 -n 20
```

### 次元圧縮後の散布図
- ax: 主成分軸（2軸を指定）
- o: 出力パス
```bash
# PC1 vs. PC2 の散布図を hogehoge.pngとして出力する
poetry run python main.py scatter data/hoge.csv -ax 1 -ax 2 -o hogehoge.png
```
