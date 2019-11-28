# localization
sem.seg.による建物検出を行う

## model 
* unet
* attention unet

# classification
画像分類タスクよる被害判定を行う
* Siamise CNN
pre postを同時に突っ込んで特徴量作成．その特徴量を使って被害分類．

* Siamise VAEで特徴量分離
  * １段目
  Siamise VAEでpre, postで変化する成分と変化しない成分を分離可能な特徴抽出器を学習．
  * ２段目
  １段目で作った特徴量抽出器で特長量を作って，その特長量を元に被害分類．
