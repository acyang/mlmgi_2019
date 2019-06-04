Ref: https://www.nature.com/articles/nature21042

# 問題描述

![fig1](https://media.nature.com/lw926/nature-assets/nature/journal/v542/n7639/images/nature21042-f1.jpg)
# 目前解法
使用K-mean(k=2)來對原子進行分類。
![fig2](https://media.nature.com/lw926/nature-assets/nature/journal/v542/n7639/images/nature21042-f2.jpg)

# 機器學習能幫上什麼忙
從影像萃取資訊來作為訓練資料集，訓練機器來進行分類，減少人工判讀的成分。

# 流程

* TrainData.mat : rawdata contain atom coordinate for training. Matlab MAT-files.

* iAtomType_New.mat : Human Label data for training data. Matlab MAT-files.

* dAtomIntensity_TrainData_New.mat : Crop local data for certain atom in training. hdf5 format.

* TestData.mat : rawdata contain atom coordinate for testing. Matlab MAT-files.

* dAtomIntensity_TestData_New.mat : Crop local data for certain atom in testing. hdf5 format.
