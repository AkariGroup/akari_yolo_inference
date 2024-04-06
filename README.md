# akari_yolo_inference

AKARIでYOLOの推論を行うレポジトリです。  
実行には、depthaiのオンラインモデルコンバータ(http://tools.luxonis.com/) で変換したYOLOのモデルファイル(.blob)とラベルファイル(.json)が必要です。  
詳細は下記のリンク先を参照ください。  
https://akarigroup.github.io/docs/source/dev/custom_object_detection/testing.html

## セットアップ
1. (初回のみ)submoduleの更新
`git submodule update --init --recursive`  

1. (初回のみ)仮想環境の作成
`python -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  

## 実行方法
`source venv/bin/activate`  
を実施後、それぞれ下記を実行。

### 物体認識
`python3 yolo.py`

引数は下記を指定可能
- `-m`, `--model`: オリジナルのYOLO認識モデル(.blob)を用いる場合にパスを指定。引数を指定しない場合、YOLO v7のCOCOデータセット学習モデルを用いる。  
- `-c`, `--config`: オリジナルのYOLO認識ラベル(.json)を用いる場合にパスを指定。引数を指定しない場合、YOLO v7のCOCOデータセット学習ラベルを用いる。  
- `-f`, `--fps`: カメラ画像の取得PFS。デフォルトは10。OAK-Dの性質上、推論の処理速度を上回る入力を与えるとアプリが異常終了しやすくなるため注意。  

### 3次元位置物体認識
物体認識結果の3次元位置を推定可能。

`python3 spatial_yolo.py`

引数は下記を指定可能
- `-m`, `--model`: オリジナルのYOLO認識モデル(.blob)を用いる場合にパスを指定。引数を指定しない場合、YOLO v7のCOCOデータセット学習モデルを用いる。  
- `-c`, `--config`: オリジナルのYOLO認識ラベル(.json)を用いる場合にパスを指定。引数を指定しない場合、YOLO v7のCOCOデータセット学習ラベルを用いる。  
- `-f`, `--fps`: カメラ画像の取得PFS。デフォルトは10。OAK-Dの性質上、推論の処理速度を上回る入力を与えるとアプリが異常終了しやすくなるため注意。  
- `-d`, `--display_camera`: この引数をつけると、RGB,depthの入力画像も表示される。  
- `-r`, `--robot_coordinate`: 3次元位置をカメラからの相対位置でなく、ロボットからの位置に変更。AKARI本体のヘッドの向きを取得して、座標変換を行うため、ヘッドの向きによらずAKARIの正面方向の位置が表示される。引数は必要なく、`-r`, `--robot_coordinate`をつけるのみで有効になる。  

### 3次元位置物体トラッキング
物体認識結果の3次元位置を推定し、トラッキングも可能。

`python3 tracking_yolo.py`

引数は下記を指定可能
- `-m`, `--model`: オリジナルのYOLO認識モデル(.blob)を用いる場合にパスを指定。引数を指定しない場合、YOLO v7のCOCOデータセット学習モデルを用いる。  
- `-c`, `--config`: オリジナルのYOLO認識ラベル(.json)を用いる場合にパスを指定。引数を指定しない場合、YOLO v7のCOCOデータセット学習ラベルを用いる。  
- `-f`, `--fps`: カメラ画像の取得PFS。デフォルトは10。OAK-Dの性質上、推論の処理速度を上回る入力を与えるとアプリが異常終了しやすくなるため注意。  
- `-d`, `--display_camera`: この引数をつけると、RGB,depthの入力画像も表示される。  
- `-r`, `--robot_coordinate`: この引数をつけると、3次元位置をカメラからの相対位置でなく、ロボットからの位置に変更。AKARI本体のヘッドの向きを取得して、座標変換を行うため、ヘッドの向きによらずAKARIの正面方向の位置が表示される。  
- `--spatial_frame`: この引数をつけると、俯瞰マップの代わりに3次元空間へのプロット図を描画することができる。ただし描画が重く、認識、画像描画の速度が低下する。
- `-disable_orbit`: この引数をつけると、俯瞰マップ上の移動軌道の表示を無効化する。  
- `log_path`: 軌道ログを保存するディレクトリを指定。この引数を指定しない場合、軌道ログは保存されない。  
