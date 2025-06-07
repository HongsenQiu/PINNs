# 圆环油膜PINN模型训练数据格式说明

## 数据格式概述

训练数据采用极坐标系统，包含以下四个主要参数：
1. 径向坐标 r (mm)
2. 周向坐标 θ (rad)
3. 压力值 p (MPa)
4. 油膜厚度 h (mm)

## 数据范围

### 几何参数
- 内径：1.8 mm (r_inner = 0.9 mm)
- 外径：3.4 mm (r_outer = 1.7 mm)
- 径向网格点数：30
- 周向网格点数：250

### 坐标范围
- r: [0.9, 1.7] mm
- θ: [0, 2π] rad

## 数据生成方法

### 采样策略
使用拉丁超立方采样（Latin Hypercube Sampling）生成训练数据点，确保数据在参数空间中均匀分布。

### 数据生成示例代码
```python
import numpy as np
from scipy.stats import qmc

def generate_training_data(n_samples=1000):
    # 定义几何参数
    r_inner = 0.9  # mm
    r_outer = 1.7  # mm
    
    # 使用拉丁超立方采样
    sampler = qmc.LatinHypercube(d=2)
    samples = sampler.random(n_samples)
    
    # 转换到极坐标范围
    r = r_inner + (r_outer - r_inner) * samples[:, 0]
    theta = 2 * np.pi * samples[:, 1]
    
    # 生成压力和厚度数据
    pressure = 1.0 - (r - r_inner) / (r_outer - r_inner)
    thickness = 0.1 * (1 + 0.5 * np.sin(theta))
    
    return r, theta, pressure, thickness
```

## 数据格式要求

### 输入数据格式
训练数据应包含以下四个数组：
1. r: 径向坐标数组，形状为 (n_samples,)
2. theta: 周向坐标数组，形状为 (n_samples,)
3. pressure: 压力值数组，形状为 (n_samples,)
4. thickness: 厚度值数组，形状为 (n_samples,)

### 数据类型
- 所有数据应转换为 torch.float32 类型
- 坐标数据需要增加一个维度用于模型输入：shape 从 (n_samples,) 变为 (n_samples, 1)

### 数据预处理
```python
# 数据转换为张量
r_tensor = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
theta_tensor = torch.tensor(theta, dtype=torch.float32).unsqueeze(1)
pressure_tensor = torch.tensor(pressure, dtype=torch.float32)
thickness_tensor = torch.tensor(thickness, dtype=torch.float32)
```

## 可视化数据格式

### 预测结果可视化
预测结果使用极坐标网格进行可视化：
- 径向方向：30个均匀分布的点
- 周向方向：250个均匀分布的点

### 可视化数据生成
```python
# 创建预测网格
r = np.linspace(r_inner, r_outer, 30)
theta = np.linspace(0, 2*np.pi, 250)
R, Theta = np.meshgrid(r, theta)

# 转换为笛卡尔坐标用于绘图
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
```

## 注意事项

1. 数据生成时应确保：
   - 压力值在入口处（r = r_inner）最大
   - 压力值在出口处（r = r_outer）最小
   - 厚度分布应考虑周向变化

2. 训练数据量建议：
   - 最小样本数：1000
   - 推荐样本数：5000-10000

3. 数据归一化：
   - 建议对压力和厚度数据进行归一化处理
   - 坐标数据保持原始物理单位 