import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)  # 输出压力和厚度
        )
        
    def forward(self, r, theta):
        x = torch.cat([r, theta], dim=1)
        return self.net(x)

def generate_training_data(n_samples=1000):
    # 生成极坐标下的训练数据
    r_inner = 0.9  # 内径1.8mm的一半
    r_outer = 1.7  # 外径3.4mm的一半
    
    # 使用拉丁超立方采样
    sampler = qmc.LatinHypercube(d=2)
    samples = sampler.random(n_samples)
    
    # 转换到极坐标范围
    r = r_inner + (r_outer - r_inner) * samples[:, 0]
    theta = 2 * np.pi * samples[:, 1]
    
    # 生成模拟的压力和厚度数据
    # 这里使用简单的解析解作为示例
    pressure = 1.0 - (r - r_inner) / (r_outer - r_inner)  # 线性压力分布
    thickness = 0.1 * (1 + 0.5 * np.sin(theta))  # 正弦变化的厚度分布
    
    return torch.tensor(r, dtype=torch.float32), torch.tensor(theta, dtype=torch.float32), \
           torch.tensor(pressure, dtype=torch.float32), torch.tensor(thickness, dtype=torch.float32)

def physics_loss(model, r, theta):
    # 计算物理损失（简化版）
    r.requires_grad_(True)
    theta.requires_grad_(True)
    
    output = model(r, theta)
    pressure, thickness = output[:, 0], output[:, 1]
    
    # 计算压力梯度
    dp_dr = torch.autograd.grad(pressure, r, grad_outputs=torch.ones_like(pressure),
                               create_graph=True)[0]
    
    # 简化的雷诺方程损失
    loss = torch.mean(dp_dr**2)
    
    return loss

def train_model(model, n_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        
        # 生成训练数据
        r, theta, p_true, h_true = generate_training_data(1000)
        
        # 模型预测
        p_pred, h_pred = model(r.unsqueeze(1), theta.unsqueeze(1)).T
        
        # 数据损失
        data_loss = torch.mean((p_pred - p_true)**2 + (h_pred - h_true)**2)
        
        # 物理损失
        phys_loss = physics_loss(model, r.unsqueeze(1), theta.unsqueeze(1))
        
        # 总损失
        loss = data_loss + 0.1 * phys_loss
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.6f}')

def visualize_results(model):
    # 创建预测网格
    r_inner, r_outer = 0.9, 1.7
    r = np.linspace(r_inner, r_outer, 30)
    theta = np.linspace(0, 2*np.pi, 250)
    R, Theta = np.meshgrid(r, theta)
    
    # 转换为笛卡尔坐标用于绘图
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    
    # 预测压力和厚度
    r_tensor = torch.tensor(R.flatten(), dtype=torch.float32).unsqueeze(1)
    theta_tensor = torch.tensor(Theta.flatten(), dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        predictions = model(r_tensor, theta_tensor)
        pressure = predictions[:, 0].reshape(R.shape)
        thickness = predictions[:, 1].reshape(R.shape)
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 压力分布
    im1 = ax1.pcolormesh(X, Y, pressure, shading='auto', cmap='viridis')
    ax1.set_title('压力分布')
    plt.colorbar(im1, ax=ax1)
    
    # 厚度分布
    im2 = ax2.pcolormesh(X, Y, thickness, shading='auto', cmap='plasma')
    ax2.set_title('厚度分布')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()

def main():
    # 创建模型
    model = PINN()
    
    # 训练模型
    print("开始训练模型...")
    train_model(model)
    
    # 可视化结果
    print("生成预测结果可视化...")
    visualize_results(model)
    print("完成！结果已保存到 prediction_results.png")

if __name__ == "__main__":
    main()
