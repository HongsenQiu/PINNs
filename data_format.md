# 油膜数据格式说明

## 数据文件格式
训练数据应保存为NumPy数组格式（.npy文件），包含以下四个文件：

1. `theta_grid.npy`: 周向角度网格
2. `r_grid.npy`: 径向距离网格
3. `pressure.npy`: 油膜压力数据
4. `thickness.npy`: 油膜厚度数据

## 数据维度要求

### 网格数据
- 周向（theta）: 250个点，范围[0, 2π]
- 径向（r）: 30个点，范围[1.8mm, 3.4mm]
- 网格形状: (30, 250)

### 物理数据
- 压力数据（pressure）: 形状(30, 250)，单位MPa
- 厚度数据（thickness）: 形状(30, 250)，单位mm

## 数据生成示例代码

```python
import numpy as np

# 生成网格
n_theta = 250
n_r = 30
r_inner = 1.8  # mm
r_outer = 3.4  # mm

theta = np.linspace(0, 2*np.pi, n_theta)
r = np.linspace(r_inner, r_outer, n_r)
theta_grid, r_grid = np.meshgrid(theta, r)

# 生成示例数据
pressure = np.random.uniform(0.1, 1.0, size=(n_r, n_theta))  # 示例压力范围：0.1-1.0 MPa
thickness = np.random.uniform(0.01, 0.1, size=(n_r, n_theta))  # 示例厚度范围：0.01-0.1 mm

# 保存数据
np.save('theta_grid.npy', theta_grid)
np.save('r_grid.npy', r_grid)
np.save('pressure.npy', pressure)
np.save('thickness.npy', thickness)
```

## 数据验证
在生成数据后，可以使用以下代码验证数据格式是否正确：

```python
import numpy as np

# 加载数据
theta_grid = np.load('theta_grid.npy')
r_grid = np.load('r_grid.npy')
pressure = np.load('pressure.npy')
thickness = np.load('thickness.npy')

# 验证维度
assert theta_grid.shape == (30, 250), "theta_grid维度错误"
assert r_grid.shape == (30, 250), "r_grid维度错误"
assert pressure.shape == (30, 250), "pressure维度错误"
assert thickness.shape == (30, 250), "thickness维度错误"

# 验证数值范围
assert np.all(r_grid >= 1.8) and np.all(r_grid <= 3.4), "r_grid范围错误"
assert np.all(theta_grid >= 0) and np.all(theta_grid <= 2*np.pi), "theta_grid范围错误"
```

## 注意事项
1. 所有数据必须使用NumPy数组格式
2. 确保数据维度完全匹配
3. 压力数据单位统一使用MPa
4. 厚度数据单位统一使用mm
5. 角度数据使用弧度制
6. 径向距离数据使用毫米（mm） 