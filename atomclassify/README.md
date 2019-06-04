Ref: https://www.nature.com/articles/nature21042

# 問題描述
現在有原子級解析度的三維影像資料，但是無法直接從影像上區分不同的原子，只能仰賴人工的判讀標記，或者是利用特定規則來進行原子的分類。
![fig1](https://media.nature.com/lw926/nature-assets/nature/journal/v542/n7639/images/nature21042-f1.jpg)
# 目前解法
使用K-mean(k=2)來對原子進行分類。
![fig2](https://media.nature.com/lw926/nature-assets/nature/journal/v542/n7639/images/nature21042-f2.jpg)

# 機器學習能幫上什麼忙
從影像萃取資訊來作為訓練資料集，訓練機器來進行分類，減少人工判讀的成分。

# 流程
1. 利用影像中強度決定各原子位置。

2. 以第i個原子為中心，挖取鄰近一定範圍的ROI(Region of Interest)，EX.(7X7X7)，作為3D影像。

3. 先以影像資訊統計的方式選出2000個可以明顯判別中心原子類別的影像，做為已標記的訓練資料集。

4. 使用3D CNN進行中心原子分類的引擎訓練。

# 檔案說明

* TrainData.mat : rawdata contain atom coordinate for training. Matlab MAT-files.

* iAtomType_New.mat : Human Label data for training data. Matlab MAT-files.

* dAtomIntensity_TrainData_New.mat : Crop local data for certain atom in training. hdf5 format.

* TestData.mat : rawdata contain atom coordinate for testing. Matlab MAT-files.

* dAtomIntensity_TestData_New.mat : Crop local data for certain atom in testing. hdf5 format.(File size exceeds the github limitation. Please go to E3 platform to download it.)
