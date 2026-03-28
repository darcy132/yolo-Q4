mv downloads/Q4-TestDataset/测试集 Q4-Dataset/test_set
mv downloads/Q4-TrainingDataset/训练集 Q4-Dataset/train_set

python convert_to_yolo.py
python stratified_split.py