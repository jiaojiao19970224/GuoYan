import numpy as np
import matplotlib.pyplot as plt

# 定义动态风险场函数
def dynamic_risk_field(x, y, x_obs, y_obs, v_obs, v, A, sigma_v, sigma_yg, alpha, L_obs, rel_v):
    numerator = np.exp(-((x - x_obs)**2 / sigma_v**2 + (y - y_obs)**2 / sigma_yg**2))
    denominator = 1 + np.exp(rel_v * (x - x_obs - alpha * L_obs * rel_v))
    risk = A * numerator / denominator
    return risk

# 定义参数
A = 1.0  # 风险场幅度
sigma_v = 5.0  # 与速度相关的标准差
sigma_yg = 5.0  # 与y坐标相关的标准差
alpha = 0.5  # 调整系数
L_obs = 10.0  # 障碍物长度
v_obs = 10.0  # 障碍物速度
v = 5.0  # 自车速度

# 计算相对速度方向
rel_v = np.sign(v_obs - v)

# 创建网格
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x, y)

# 计算风险场
risk_field = dynamic_risk_field(X, Y, 50, 50, v_obs, v, A, sigma_v, sigma_yg, alpha, L_obs, rel_v)

# 绘制风险场
plt.figure(figsize=(8, 8))
plt.contourf(X, Y, risk_field, levels=50, cmap='viridis')
plt.colorbar(label='Risk Field')
plt.scatter(50, 50, color='red')  # 标记障碍物位置
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.title('Dynamic Risk Field')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()