import numpy as np

# 填入你刚才使用的动捕文件路径
file_path = './data/motion/dance1_subject1.npz'

# 1. 加载 npz 文件
data = np.load(file_path, allow_pickle=True)

# 2. 打印文件里包含了哪些数据变量
print(">>> 文件中包含的变量 (Keys):")
print(data.files)
print("-" * 40)

# 3. 遍历打印每一个变量的维度 (Shape)
for key in data.files:
    array = data[key]
    # 有些变量可能是纯标量或者字符串，加个判断防止报错
    if hasattr(array, 'shape'):
        print(f"变量 '{key}' 的维度是: {array.shape}")
    else:
        print(f"变量 '{key}' 没有 shape 属性，值大概是: {array}")

print("-" * 40)

# 4. 重点诊断我们的核心疑问
if 'joint_pos' in data.files:
    shape = data['joint_pos'].shape
    print(f"💡 破案线索: 'joint_pos' (关节角度) 的维度是 {shape}")
    print(f"   -> 总共有 {shape[0]} 帧动作")
    print(f"   -> 机器人的动作自由度 (DoF) 是 {shape[1]}！")