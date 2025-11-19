import random

# 读取所有行
with open('all.txt', 'r') as f:
    lines = f.readlines()

# 打乱顺序
random.shuffle(lines)

# 按9:1划分
num_total = len(lines)
num_train = int(num_total * 0.9)

train_lines = lines[:num_train]
val_lines = lines[num_train:]

# 写入 train.txt 和 val.txt
with open('train.txt', 'w') as f:
    f.writelines(train_lines)

with open('val.txt', 'w') as f:
    f.writelines(val_lines)

print(f"Total: {num_total}, Train: {len(train_lines)}, Val: {len(val_lines)}")