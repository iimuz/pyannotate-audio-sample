---
title: pyannoteを利用した話者分離サンプル
date: 2023-10-11
lastmod: 2023-10-11
---

## 概要

[pyannote-audio](https://github.com/pyannote/pyannote-audio)を利用した話者分離のサンプルです。

## 実行方法

```sh
# python仮想環境の作成
python -m venv .venv
source .venv/bin/activate
pip install -e .

# モデルのダウンロード
python src/download_model.py

# 話者分離の実行
python src/annote.py
```

## Tips

### モデルのダウンロード

公式に記載の方法では Token を取得するだけでは動作しない。
下記の記事にあるようにモデルをダウンロードするスクリプトを利用してモデルをダウンロード後、config.yaml と pytorch_model.bin をコピーして書き換えることで動作することを確認した。
ダウンロードして huggingface の cache ディレクトリにある状態だと正常に動作しなかった。

- [pyannote.audio がメジャーアップデートし 3.0.0 がリリースされました](https://dev.classmethod.jp/articles/pyannote-audio-v3/)

### wav ファイルの切り出し

下記コマンドを実行することで特定区間の wav に切り出すことができる。
対象のファイルが大きくて時間がかかる場合は、下記コマンドで切り出すようにすると良い。

```sh
# 開始30秒(-ss 30)地点から15秒(-t 15)の切り出しを行う
ffmpeg -i hoge.wav -ss 30 -t 15 -acodec copy hoge_cut.wav
```
