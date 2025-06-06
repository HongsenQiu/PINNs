# PINNs 油膜压力与厚度预测

本项目基于物理信息神经网络（PINNs）实现极坐标下油膜压力与厚度的预测。

## 项目简介
- 使用 PINNs 方法求解油膜雷诺方程，预测极坐标区域内的压力和厚度分布。
- 采用 `pinnstorch` 框架，结合 PyTorch 实现。
- 支持自定义边界条件和训练数据生成。

## 依赖环境
- Python 3.8+
- torch
- numpy
- matplotlib
- pinnstorch

建议使用虚拟环境（如 venv 或 conda）管理依赖。

安装依赖示例：
```powershell
pip install torch numpy matplotlib pinnstorch
```

## 运行方法
1. 克隆或下载本项目到本地。
2. 安装依赖。
3. 运行主程序：
   ```powershell
   python main.py
   ```
4. 训练完成后，预测结果会保存在 `prediction_results.png`。

## 主要文件说明
- `main.py`：主程序，包含数据生成、模型定义、训练与预测流程。
- `prediction_results.png`：模型预测的压力与厚度分布图。
- `pyproject.toml`、`uv.lock`：依赖管理文件。

## 结果展示
运行结束后将在当前目录生成 `prediction_results.png`，展示预测的压力分布和厚度分布。

## 参考
- [PINNs 论文](https://www.sciencedirect.com/science/article/pii/S0021999118307125)
- [pinnstorch 文档](https://github.com/your-pinnstorch-repo)
