import torch
import numpy as np
import matplotlib.pyplot as plt
from pinnstorch.models import PINN
from pinnstorch.domain import Domain, BoundaryCondition

def generate_training_data(n_theta=250, n_r=30):
    # 生成极坐标网格
    r_inner = 1.8  # mm
    r_outer = 3.4  # mm
    theta = np.linspace(0, 2*np.pi, n_theta)
    r = np.linspace(r_inner, r_outer, n_r)
    
    # 生成网格点
    theta_grid, r_grid = np.meshgrid(theta, r)
    
    # 生成随机压力和厚度数据
    pressure = np.random.uniform(0.1, 1.0, size=(n_r, n_theta))
    thickness = np.random.uniform(0.01, 0.1, size=(n_r, n_theta))
    
    return theta_grid, r_grid, pressure, thickness

def oil_film_pde(model, x):
    """
    油膜雷诺方程的PDE约束
    """
    r, theta = x[:, 0:1], x[:, 1:2]
    
    # 计算压力和厚度对r和theta的导数
    p = model(x)
    h = p[:, 1:2]  # 厚度
    p = p[:, 0:1]  # 压力
    
    # 计算一阶导数
    dp_dr = torch.autograd.grad(p, r, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dp_dtheta = torch.autograd.grad(p, theta, grad_outputs=torch.ones_like(p), create_graph=True)[0]
    dh_dr = torch.autograd.grad(h, r, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    dh_dtheta = torch.autograd.grad(h, theta, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    
    # 计算二阶导数
    d2p_dr2 = torch.autograd.grad(dp_dr, r, grad_outputs=torch.ones_like(dp_dr), create_graph=True)[0]
    d2p_dtheta2 = torch.autograd.grad(dp_dtheta, theta, grad_outputs=torch.ones_like(dp_dtheta), create_graph=True)[0]
    
    # 简化的雷诺方程
    residual = d2p_dr2 + (1/r**2)*d2p_dtheta2 + (1/r)*dp_dr - 12*eta*(dh_dr/r + dh_dtheta/(r**2))
    
    return residual

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 生成训练数据
    theta_grid, r_grid, pressure, thickness = generate_training_data()
    
    # 定义计算域
    domain = Domain(
        x_min=[1.8, 0],  # r_min, theta_min
        x_max=[3.4, 2*np.pi],  # r_max, theta_max
        n_points=[30, 250]  # n_r, n_theta
    )
    
    # 定义边界条件
    bc_inner = BoundaryCondition(
        x_min=[1.8, 0],
        x_max=[1.8, 2*np.pi],
        n_points=[1, 250],
        value=1.0  # 内壁压力入口
    )
    
    bc_outer = BoundaryCondition(
        x_min=[3.4, 0],
        x_max=[3.4, 2*np.pi],
        n_points=[1, 250],
        value=0.0  # 外壁压力出口
    )
    
    # 创建PINN模型
    model = PINN(
        input_dim=2,  # r, theta
        output_dim=2,  # pressure, thickness
        hidden_layers=[20, 20, 20, 20],
        activation='tanh'
    )
    
    # 设置PDE约束
    model.set_pde(oil_film_pde)
    
    # 设置边界条件
    model.set_boundary_conditions([bc_inner, bc_outer])
    
    # 训练模型
    model.train(
        epochs=1000,
        learning_rate=0.001,
        batch_size=1000
    )
    
    # 预测结果
    r = torch.tensor(r_grid.flatten(), dtype=torch.float32).reshape(-1, 1)
    theta = torch.tensor(theta_grid.flatten(), dtype=torch.float32).reshape(-1, 1)
    x = torch.cat([r, theta], dim=1)
    
    with torch.no_grad():
        predictions = model(x)
    
    # 重塑预测结果
    pressure_pred = predictions[:, 0].reshape(r_grid.shape)
    thickness_pred = predictions[:, 1].reshape(r_grid.shape)
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 压力分布
    im1 = ax1.pcolormesh(theta_grid, r_grid, pressure_pred, shading='auto')
    ax1.set_title('预测压力分布')
    plt.colorbar(im1, ax=ax1)
    
    # 厚度分布
    im2 = ax2.pcolormesh(theta_grid, r_grid, thickness_pred, shading='auto')
    ax2.set_title('预测厚度分布')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()
    
    print("训练完成，结果已保存为 prediction_results.png")

if __name__ == "__main__":
    main()
