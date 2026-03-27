import numpy as np

counts = [1000, 47, 1330, 1631, 702, 492, 256, 285]

# 对数平滑，避免小样本权重过大导致训练不稳定
weights_log = np.log(np.max(counts) / np.array(counts)) + 1

# 转换为 Python float 并输出
weights_float = [round(float(w), 4) for w in weights_log]
print("对数平滑权重:", weights_float)