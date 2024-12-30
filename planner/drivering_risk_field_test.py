import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap




import matplotlib.cm as cm

# 读取CSV数据
data = pd.read_csv(r"F:\Users\yq\Desktop\scope_adjust\01_tracks2.csv")


# 定义静态风险场函数
def static_risk_field(x, y, x_obs, y_obs, A, kx, ky, L_obs, W_obs):
    sigma_xg = kx * L_obs
    sigma_yg = ky * W_obs
    return A * np.exp(-((x - x_obs) ** 2 / (2 * sigma_xg ** 2)) - ((y - y_obs) ** 2 / (2 * sigma_yg ** 2)))


# 定义动态风险场函数
# def dynamic_risk_field(x, y, x_obs, y_obs, v_obs, v, A, kv, alpha, L_obs_relv):
#     sigma_v = kv * abs(v_obs - v)
#     rel_v = np.sign(v_obs - v)
#     return A * np.exp(-((x - x_obs) ** 2 / (2 * sigma_v ** 2)) - ((y - y_obs) ** 2 / (2 * sigma_v * L_obs_relv) ** 2)) * \
#         (1 + np.exp(rel_v * (x - x_obs - alpha * L_obs_relv)))

def dynamic_risk_field(x, y, x_obs, y_obs, v_obs, v, A, kv, alpha, L_obs):
    epsilon = 1e-10  # 添加一个小的正数以避免除以零
    sigma_v = kv * abs(v_obs - v) + epsilon
    rel_v = np.sign(v_obs - v)
    return A * np.exp(-((x - x_obs) ** 2 / (2 * sigma_v ** 2)) - ((y - y_obs) ** 2 / (2 * sigma_v * (L_obs * rel_v + epsilon)) ** 2)) * \
        (1 + np.exp(rel_v * (x - x_obs - alpha * L_obs * rel_v)))


# 初始化绘图
fig, ax = plt.subplots(figsize=(10, 6))

# 获取所有唯一的帧号
frames = data['frame'].unique()

# 固定坐标轴范围
min_x = data['x'].min() - 10
max_x = data['x'].max() + 10
min_y = data['y'].min() - 10
max_y = data['y'].max() + 10
ax.set_xlim(min_x, max_x)
ax.set_ylim(min_y, max_y)

###################################################################
# 创建一个对比度更高的颜色映射
colors = [(0, 0, 0, 0), (0, 0, 1, 1), (0, 1, 0, 1), (1, 0, 0, 1)]  # 透明, 蓝色, 绿色, 红色
cmap_custom = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

# 设置自定义颜色映射为默认颜色映射
plt.rcParams['image.cmap'] = cmap_custom


###################################################################

def update(frame):
    ax.cla()  # 清除旧的图层，仅更新数据
    ax.set_xlim(min_x, max_x)  # 确保坐标轴固定
    ax.set_ylim(min_y, max_y)  # 确保坐标轴固定
    frame_data = data[data['frame'] == frame]

    ax.set_title(f"Frame {frame}")

    for _, row in frame_data.iterrows():
        # 绘制车身
        car_center_x = row['x']
        car_center_y = row['y']
        car_width = row['width']  # 使用 'width' 列代替 'L_obs'
        car_height = row['height']  # 使用 'height' 列代替 'W_obs'
        car_rect = plt.Rectangle((car_center_x - car_width / 2, car_center_y - car_height / 2), car_width, car_height,
                                 fill=True, color='gray', alpha=0.8)
        ax.add_patch(car_rect)

        # 计算风险场
        X, Y = np.meshgrid(np.linspace(row['x'] - 50, row['x'] + 50, 100),
                           np.linspace(row['y'] - 50, row['y'] + 50, 100))
        Z_static = static_risk_field(X, Y, row['x'], row['y'], 1, 1, 1, row['width'], row['height'])
        Z_dynamic = dynamic_risk_field(X, Y, row['x'], row['y'], row['xVelocity'], row['xVelocity'], 1, 0.1, 0.1,
                                       row['width'])
        Z = Z_static + Z_dynamic
        ax.imshow(Z, extent=(row['x'] - 50, row['x'] + 50, row['y'] - 50, row['y'] + 50), origin='lower',
                  cmap=cmap_custom, alpha=0.8)


# def update(frame):
#     ax.cla()  # 清除旧的图层，仅更新数据
#     ax.set_xlim(min_x, max_x)  # 确保坐标轴固定
#     ax.set_ylim(min_y, max_y)  # 确保坐标轴固定
#     frame_data = data[data['frame'] == frame]
#
#     ax.set_title(f"Frame {frame}")
#
#     for _, row in frame_data.iterrows():
#         X, Y = np.meshgrid(np.linspace(row['x'] - 50, row['x'] + 50, 100),
#                            np.linspace(row['y'] - 50, row['y'] + 50, 100))
#         Z_static = static_risk_field(X, Y, row['x'], row['y'], 1, 1, 1, row['width'], row['height'])
#         Z_dynamic = dynamic_risk_field(X, Y, row['x'], row['y'], row['xVelocity'], row['xVelocity'], 1, 0.1, 0.1,
#                                        row['width'])
#         Z = Z_static + Z_dynamic
#         ax.imshow(Z, extent=(row['x'] - 50, row['x'] + 50, row['y'] - 50, row['y'] + 50), origin='lower',
#                   cmap=cmap_custom, alpha=0.8)


# 创建动画
ani = FuncAnimation(fig, update, frames=frames, interval=50, repeat=False)

plt.show()