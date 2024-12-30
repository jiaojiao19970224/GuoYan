import time
from datetime import datetime

'''
全局路径规划类
'''
import networkx as nx
import numpy as np
from enum import Enum
import carla

def Vector_fun(loc_1: carla.Location, loc_2: carla.Location):
    delt_x = loc_2.x - loc_1.x
    delt_y = loc_2.y - loc_1.y
    delt_z = loc_2.z - loc_1.z
    norm = np.linalg.norm([delt_x, delt_y, delt_z]) + np.finfo(float).eps  # eps取非负的最小值，为了不让norm为零
    return np.round([delt_x / norm, delt_y / norm, delt_z / norm], 4)  # 保留4位小数


def waypoint_list_2_target_path(pathway):
    target_path = []
    w = None  # type: carla.Waypoint
    xy_list_ori = []
    for w in pathway:
        x = w[0].transform.location.x
        y = w[0].transform.location.y
        xy_list_ori.append((x, y))


    theta_list, kappa_list = cal_heading_kappa(xy_list_ori)  # 包含frenet曲线上每一点的坐标[(x0,y0), (x1, y1), ...]
    # self._target_path = smooth_reference_line(xy_list_ori)  # 对生成的原始轨迹进行平滑,这里只是做了个实验
    for i in range(len(theta_list)):
        target_path.append((xy_list_ori[i][0], xy_list_ori[i][1], theta_list[i], kappa_list[i]))
        target_path.append((xy_list_ori[i][0], xy_list_ori[i][1], theta_list[i], kappa_list[i]))
    return target_path


def find_match_points(xy_list: list, frenet_path_node_list: list, is_first_run: bool, pre_match_index: int):
    i = -1  # 提前定义一个变量用于遍历曲线上的点
    input_xy_length = len(xy_list)
    frenet_path_length = len(frenet_path_node_list)

    match_point_index_list = np.zeros(input_xy_length, dtype="int32")
    project_node_list = []  # 最终长度和input_xy_length应该相同

    if is_first_run is True:
        for index_xy in range(input_xy_length):  # 为每一个点寻找匹配点
            x, y = xy_list[index_xy]
            start_index = 0
            # 用increase_count记录distance连续增大的次数，避免多个局部最小值的干扰
            increase_count = 0
            min_distance = float("inf")
            # 确定匹配点
            for i in range(start_index, frenet_path_length):
                frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
                # 计算(x,y)与(frenet_node_x, frenet_node_y)之间的距离
                distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
                if distance < min_distance:
                    min_distance = distance  # 保留最小值
                    match_point_index_list[index_xy] = i
                    increase_count = 0
                else:
                    increase_count += 1
                    if increase_count >= 50:  # 向后50个点还没找到的话，说明当前最小值点及时最优的
                        # 第一次运行阈值较大，是为了保证起始点匹配的精确性
                        break
            # 通过匹配点确定投影点
            match_point_index = match_point_index_list[0]
            x_m, y_m, theta_m, k_m = frenet_path_node_list[match_point_index]
            d_v = np.array([x - x_m, y - y_m])
            tou_v = np.array([np.cos(theta_m), np.sin(theta_m)])
            ds = np.dot(d_v, tou_v)
            r_m_v = np.array([x_m, y_m])
            # 根据公式计算投影点的位置信息
            x_r, y_r = r_m_v + ds * tou_v  # 计算投影点坐标
            theta_r = theta_m + k_m * ds  # 计算投影点在frenet曲线上切线与X轴的夹角
            k_r = k_m  # 投影点在frenet曲线处的曲率
            # 将结果打包放入缓存区
            project_node_list.append((x_r, y_r, theta_r, k_r))

    else:
        for index_xy in range(input_xy_length):  # 为每一个点寻找匹配点
            x, y = xy_list[index_xy]
            start_index = pre_match_index

            # TODO 判断帧与帧之间的障碍物是否为同一个

            # 用increase_count记录distance连续增大的次数，避免多个局部最小值的干扰
            # TODO 判断是否有上个周期的index结果，没有的话start_index=1, increase_count_limit=50
            increase_count_limit = 5
            increase_count = 0
            # 上个周期匹配点坐标
            pre_match_point_xy = [frenet_path_node_list[start_index][0], frenet_path_node_list[start_index][1]]
            pre_match_point_theta_m = frenet_path_node_list[start_index][2]
            # 上个匹配点在曲线上的切向向量
            pre_match_point_direction = np.array([np.cos(pre_match_point_theta_m), np.sin(pre_match_point_theta_m)])
            # 计算上个匹配点指向当前(x, y）的向量
            pre_match_to_xy_v = np.array([x - pre_match_point_xy[0], y - pre_match_point_xy[1]])
            # 计算pre_match_to_xy_v在pre_match_point_direction上的投影，用于判断遍历方向
            flag = np.dot(pre_match_to_xy_v, pre_match_point_direction)  # 大于零正反向遍历，反之，反方向遍历

            min_distance = float("inf")
            if flag > 0:  # 正向遍历
                for i in range(start_index, frenet_path_length):
                    frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
                    # 计算（x,y) 与 （frenet_node_x, frenet_node_y） 之间的距离
                    distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
                    if distance < min_distance:
                        min_distance = distance  # 保留最小值
                        match_point_index_list[index_xy] = i
                        increase_count = 0
                    else:
                        increase_count += 1
                        if increase_count >= increase_count_limit:  # 为了加快速度，这里阈值为5，第一个周期是不同的，向后5个点还没找到的话，说明当前最小值点及时最优的
                            break
            else:  # 反向遍历
                for i in range(start_index, -1, -1):  # range(start,end,step)，其中start为闭，end为开，-1为步长
                    frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
                    # 计算（x,y) 与 （frenet_node_x, frenet_node_y） 之间的距离
                    distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
                    if distance < min_distance:
                        min_distance = distance  # 保留最小值
                        match_point_index_list[index_xy] = i
                        increase_count = 0
                    else:
                        increase_count += 1
                        if increase_count >= increase_count_limit:
                            break
            # 通过匹配点确定投影点
            match_point_index = match_point_index_list[0]
            x_m, y_m, theta_m, k_m = frenet_path_node_list[match_point_index]
            d_v = np.array([x - x_m, y - y_m])
            tou_v = np.array([np.cos(theta_m), np.sin(theta_m)])
            ds = np.dot(d_v, tou_v)
            r_m_v = np.array([x_m, y_m])
            # 根据公式计算投影点的坐标信息
            x_r, y_r = r_m_v + ds * tou_v
            theta_r = theta_m + k_m * ds
            k_r = k_m
            # 将结果打包放入缓存区
            project_node_list.append((x_r, y_r, theta_r, k_r))

    return list(match_point_index_list), project_node_list


def cal_heading_kappa(frenet_path_xy_list: list):
    dx_ = []
    dy_ = []
    for i in range(len(frenet_path_xy_list) - 1):
        dx_.append(frenet_path_xy_list[i + 1][0] - frenet_path_xy_list[i][0])
        dy_.append(frenet_path_xy_list[i + 1][1] - frenet_path_xy_list[i][1])
    # 计算theta,切线方向角
    # 由于n个点差分得到的只有n-1个差分结果，所以要在首尾添加重复单元来近似求每个节点的dx,dy
    dx_pre = [dx_[0]] + dx_  # 向前补dx_的第一位
    dx_aft = dx_ + [dx_[-1]]  # 向后补dx_的最后一位
    dx = (np.array(dx_pre) + np.array(dx_aft)) / 2

    dy_pre = [dy_[0]] + dy_
    dy_aft = dy_ + [dy_[-1]]
    dy = (np.array(dy_pre) + np.array(dy_aft)) / 2
    theta = np.arctan2(dy, dx)  # np.arctan2会将角度限制在（-pi, pi）之间
    # 计算曲率
    d_theta_ = np.diff(theta)  # 差分计算
    d_theta_pre = np.insert(d_theta_, 0, d_theta_[0])
    d_theta_aft = np.insert(d_theta_, -1, d_theta_[-1])
    d_theta = np.sin((d_theta_pre + d_theta_aft) / 2)  # 认为d_theta是个小量，用sin(d_theta)代替d_theta,避免多值性
    ds = np.sqrt(dx ** 2 + dy ** 2)
    k = d_theta / ds

    return list(theta), list(k)


def sampling(match_point_index: int, frenet_path_node_list: list, back_length=10, forward_length=50):
    local_frenet_path = []
    # back_length = 10
    # forward_length = 40
    length_sum = back_length + forward_length
    if match_point_index < back_length:
        back_length = match_point_index
        forward_length = length_sum - back_length

    if (len(frenet_path_node_list) - match_point_index) - 1 < forward_length:
        forward_length = len(frenet_path_node_list) - match_point_index - 1
        back_length = length_sum - forward_length

    # 返回这个区间的node
    local_frenet_path = frenet_path_node_list[match_point_index - back_length: match_point_index] \
                        + frenet_path_node_list[match_point_index: match_point_index + forward_length + 1]

    return local_frenet_path


def smooth_reference_line(local_frenet_path_xy: list,
                          w_cost_smooth=0.4, w_cost_length=0.3, w_cost_ref=0.3,
                          x_thre=0.2, y_thre=0.2):
    n = len(local_frenet_path_xy)  # 该模块是对参考线输出进行处理的时候就是处理181个点
    x_ref = np.zeros(shape=(2 * n, 1))  # [x_ref0, y_ref0, x_ref1, y_ref1, ...]' 输入坐标构成的坐标矩阵， (2*n, 1)
    lb = np.zeros(shape=(2 * n, 1))
    ub = np.zeros(shape=(2 * n, 1))
    for i in range(n):
        x_ref[2 * i] = local_frenet_path_xy[i][0]  # x
        x_ref[2 * i + 1] = local_frenet_path_xy[i][1]  # y
        # 确定上下边界
        lb[2 * i] = local_frenet_path_xy[i][0] - x_thre
        lb[2 * i + 1] = local_frenet_path_xy[i][1] - y_thre
        ub[2 * i] = local_frenet_path_xy[i][0] + x_thre
        ub[2 * i + 1] = local_frenet_path_xy[i][1] + y_thre

    A1 = np.zeros(shape=(2 * n - 4, 2 * n))
    for i in range(n - 2):
        A1[2 * i][2 * i + 0] = 1
        # A1[2 * i][2 * i + 1] = 0
        A1[2 * i][2 * i + 2] = -2
        # A1[2 * i][2 * i + 3] = 0
        A1[2 * i][2 * i + 4] = 1
        # A1[2 * i][2 * i + 5] = 0

        # A1[2 * i + 1][2 * i + 0] = 0
        A1[2 * i + 1][2 * i + 1] = 1
        # A1[2 * i + 1][2 * i + 2] = 0
        A1[2 * i + 1][2 * i + 3] = -2
        # A1[2 * i + 1][2 * i + 4] = 0
        A1[2 * i + 1][2 * i + 5] = 1

    A2 = np.zeros(shape=(2 * n - 2, 2 * n))
    for i in range(n - 1):
        A2[2 * i][2 * i + 0] = 1
        # A2[2 * i][2 * i + 1] = 0
        A2[2 * i][2 * i + 2] = -1
        # A2[2 * i][2 * i + 3] = 0

        # A2[2 * i + 1][2 * i + 0] = 0
        A2[2 * i + 1][2 * i + 1] = 1
        # A2[2 * i + 1][2 * i + 2] = 0
        A2[2 * i + 1][2 * i + 3] = -1

    A3 = np.identity(2 * n)  # 对角阵
    H = 2 * (w_cost_smooth * np.dot(A1.transpose(), A1) +
             w_cost_length * np.dot(A2.transpose(), A2) +
             w_cost_ref * A3)

    f = -2 * w_cost_ref * x_ref
    # 将约束转化为矩阵形式
    G = np.concatenate((np.identity(2 * n), -np.identity(2 * n)))  # （4n, 2n）
    h = np.concatenate((ub, -lb))  # (4n, 1)
    cvxopt.solvers.options['show_progress'] = False  # 程序没有问题之后不再输出中间过程
    # 计算时要将输入转化为cvxopt.matrix
    # 该方法返回值是一个字典类型，包含了很多的参数，其中x关键字对应的是优化后的解
    res = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f), G=cvxopt.matrix(G), h=cvxopt.matrix(h))
    local_path_xy_opt = []
    for i in range(0, len(res['x']), 2):
        local_path_xy_opt.append((res['x'][i], res['x'][i + 1]))
    theta_list, k_list = cal_heading_kappa(local_path_xy_opt)
    x_y_theta_kappa_list = []
    for i in range(len(local_path_xy_opt)):
        x_y_theta_kappa_list.append(local_path_xy_opt[i] + (theta_list[i], k_list[i]))
    return x_y_theta_kappa_list


def match_projection_points(xy_list: list, frenet_path_node_list: list):
    input_xy_length = len(xy_list)
    frenet_path_length = len(frenet_path_node_list)

    match_point_index_list = np.zeros(input_xy_length, dtype="int32")
    project_node_list = []  # 最终长度和input_xy_length应该相同

    for index_xy in range(input_xy_length):  # 为每一个点寻找匹配点
        x, y = xy_list[index_xy][0], xy_list[index_xy][1]
        start_index = 0
        # 用increase_count记录distance连续增大的次数，避免多个局部最小值的干扰
        increase_count = 0
        min_distance = float("inf")
        # 确定匹配点
        for i in range(start_index, frenet_path_length):
            frenet_node_x, frenet_node_y, _, _ = frenet_path_node_list[i]
            # 计算（x,y) 与 （frenet_node_x, frenet_node_y） 之间的距离
            distance = math.sqrt((frenet_node_x - x) ** 2 + (frenet_node_y - y) ** 2)
            if distance < min_distance:
                min_distance = distance  # 保留最小值
                match_point_index_list[index_xy] = i
                increase_count = 0
            else:
                increase_count += 1
                if increase_count >= 50:  # 向后50个点还没找到的话，说明当前最小值点及时最优的
                    # 第一次运行阈值较大，是为了保证起始点匹配的精确性
                    break
        # 通过匹配点确定投影点
        match_point_index = match_point_index_list[0]
        x_m, y_m, theta_m, k_m = frenet_path_node_list[match_point_index]
        d_v = np.array([x - x_m, y - y_m])
        tou_v = np.array([np.cos(theta_m), np.sin(theta_m)])
        ds = np.dot(d_v, tou_v)
        r_m_v = np.array([x_m, y_m])
        # 根据公式计算投影点的位置信息
        x_r, y_r = r_m_v + ds * tou_v  # 计算投影点坐标
        theta_r = theta_m + k_m * ds  # 计算投影点在frenet曲线上切线与X轴的夹角
        k_r = k_m  # 投影点在frenet曲线处的曲率
        # 将结果打包放入缓存区
        project_node_list.append((x_r, y_r, theta_r, k_r))

    return list(match_point_index_list), project_node_list


def cal_projection_s_fun(local_path_opt: list, match_index_list: list, xy_list: list, s_map: list):
    projection_s_list = []
    for i in range(len(match_index_list)):
        x, y, theta, kappa = local_path_opt[match_index_list[i]]
        d_v = np.array([xy_list[i][0] - x, xy_list[i][1] - y])  # 匹配点指向给定点的向量
        tou_v = np.array([math.cos(theta), math.sin(theta)])  # 切线方向单位向量
        projection_s_list.append(s_map[match_index_list[i]] + np.dot(d_v, tou_v))  # np.dot(d_v, tou_v)即ds大小,有正负号的

    return projection_s_list


def cal_s_map_fun(local_path_opt: list, origin_xy: tuple):
    # 计算以车辆当前位置投影点为起点的s_map
    origin_match_index, _ = match_projection_points([origin_xy], local_path_opt)
    # 通过车辆定位位置，计算其在参考线上的匹配点索引和投影点信息，match_projection_points处理的是一系列点的列表，
    # 因此输入要为列表形式，输出也是列表形式，但是里面只有一个元素(当前位置)，因此索引第一位就行了
    origin_match_index = origin_match_index[0]
    ref_s_map = [0]
    # 先计算以参考线起点为起点的ref_s_map
    for i in range(1, len(local_path_opt)):
        s = math.sqrt((local_path_opt[i][0] - local_path_opt[i - 1][0]) ** 2
                      + (local_path_opt[i][1] - local_path_opt[i - 1][1]) ** 2) + ref_s_map[-1]
        ref_s_map.append(s)
    # print("ref_s_map", ref_s_map)
    # 然后算出在车辆当前位置投影点相对于参考线起点的s, 记为s0
    s0 = cal_projection_s_fun(local_path_opt, [origin_match_index], [origin_xy], ref_s_map)
    # ref_s_map 每一项都减去s0，这样就得到了所有匹配点相对于车辆投影点的s映射
    s_map = np.array(ref_s_map) - s0[0]
    return list(s_map)


def cal_s_l_fun(obs_xy_list: list, local_path_opt: list, s_map: list):
    # 计算这些障碍物点在当前参考线中匹配点的索引和投影点信息
    match_index_list, projection_list = match_projection_points(obs_xy_list, local_path_opt)

    s_list = cal_projection_s_fun(local_path_opt, match_index_list, obs_xy_list, s_map)  # 得到障碍点的s

    # 下面是计算l, 这里计算l和下面一个计算导数的函数有重复，以后可以考虑删除
    l_list = []

    for i in range(len(obs_xy_list)):
        pro_x, pro_y, theta, kappa = projection_list[i]  # 投影点的信息
        n_r = np.array([-math.sin(theta), math.cos(theta)])  # 投影点的单位法向量
        x, y = obs_xy_list[i][0], obs_xy_list[i][1]  # 待投影的位置
        r_h = np.array([x, y])  # 车辆实际位置的位矢
        r_r = np.array([pro_x, pro_y])  # 投影点的位矢
        # 核心公式: l*n_r = r_h - r_r
        # TODO 将左手系改为右手系，使得SL图为常见方向
        l_list.append(np.dot(r_h - r_r, n_r))  # *UE4定义的是左手系，所以在车辆左侧的为负值(SL图是反的)*

    return s_list, l_list


def cal_s_l_deri_fun(xy_list: list, V_xy_list: list, a_xy_list: list, local_path_xy_opt: list, origin_xy: tuple):
    # 计算这些障碍物点在当期参考线中匹配点的索引和投影点信息
    match_index_list, projection_list = match_projection_points(xy_list, local_path_xy_opt)

    l_list = []  # store l
    dl_list = []  # store the derivative of l, dl/dt
    ds_list = []  # store the derivative of s, ds/dt
    ddl_list = []  # store the  second order derivative of l, d(dl/dt)/dt
    l_ds_list = []  # store the arc differential of l, dl/ds
    dds_list = []  # store the second order derivative of s, d(ds/dt)/dt
    l_dds_list = []  # store the second order arc differential of l, d(dl/ds)/ds
    for i in range(len(xy_list)):
        x, y, theta, kappa = projection_list[i]  # 投影点的信息
        nor_r = np.array([-math.sin(theta), math.cos(theta)])  # 投影点的单位法向量  **************************************
        tou_r = np.array([math.cos(theta), math.sin(theta)])  # 投影点的单位切向量
        r_h = np.array([origin_xy[0], origin_xy[1]])  # 车辆实际位置的位矢
        r_r = np.array([x, y])  # 投影点的位置矢

        """1.calculate l"""
        l = np.dot(r_h - r_r, nor_r)
        l_list.append(l)

        """2.calculate dl"""
        Vx, Vy = V_xy_list[i]
        V_h = np.array([Vx, Vy])  # 速度矢量
        dl = np.dot(V_h, nor_r)
        dl_list.append(dl)  # l对时间的导数

        """3.计算s对时间的导数"""
        ds = np.dot(V_h, tou_r) / (1 - kappa * l_list[i])
        ds_list.append(ds)

        """4.计算l对时间的二阶导数"""
        ax, ay = a_xy_list[i]
        a_h = np.array([ax, ay])
        ddl = np.dot(a_h, nor_r) - kappa * (1 - kappa * l) * (ds ** 2)
        ddl_list.append(ddl)

        """5.calculate the arc differential of l"""
        #  向量法做cartesian与frenet的转换要更简单，但是也有缺点，向量法必须依赖速度加速度
        #  l' = l_dot/s_dot 但是如果s_dot = 0 此方法就失效了
        if abs(ds) < 1e-6:
            l_ds = 0
        else:
            l_ds = dl_list[i] / ds
        l_ds_list.append(l_ds)

        """6.calculate the second order derivative of s"""
        ax, ay = a_xy_list[i]
        a_h = np.array([ax, ay])
        kappa_ds = 0  # dk/ds, for simplicity, make it to be zero
        dds = (np.dot(a_h, tou_r) + 2 * (ds ** 2 * kappa * l_ds) + ds ** 2 * kappa_ds * l) / (1 - kappa * l)
        dds_list.append(dds)

        """7.the second order derivative of s"""
        if abs(ds) < 1e-6:
            l_dds = 0
        else:
            l_dds = (ddl - l_ds * dds) / (ds ** 2)
        l_dds_list.append(l_dds)

    return l_list, dl_list, ds_list, ddl_list, l_ds_list, dds_list, l_dds_list


def predict_block(ego_vehicle, ts):
    vehicle_loc = ego_vehicle.get_location()
    x, y = vehicle_loc.x, vehicle_loc.y
    fi = ego_vehicle.get_transform().rotation.yaw * (math.pi / 180)  # 车身横摆角（弧度制），车头朝向和x轴的夹角
    V = ego_vehicle.get_velocity()  # 航向角是速度方向与x轴夹角
    V_length = math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)  # 速度标量
    beta = math.atan2(V.y, V.x) - fi  # 质心侧偏角，速度方向和车头朝向之间的夹角
    # print("beta", beta, "fi", fi)
    V_y = V_length * math.sin(beta)  # 车速在车身坐标系下的分量
    V_x = V_length * math.cos(beta)
    # print("Vx", Vx, "Vy", Vy)
    x = x + V_x * ts * math.cos(fi) - V_y * ts * math.sin(fi)  # ?
    y = y + V_y * ts * math.cos(fi) + V_x * ts * math.sin(fi)
    fi_dao = ego_vehicle.get_angular_velocity().z * (math.pi / 180)
    fi = fi + fi_dao * ts

    return x, y, fi


def predict_block_based_on_frenet(vehicle_loc, vehicle_velocity, local_frenet_path_opt, cur_path_s, cur_path_l, ts=0.1):
    speed = math.sqrt(vehicle_velocity.x * vehicle_velocity.x +
                      vehicle_velocity.y * vehicle_velocity.y +
                      vehicle_velocity.z * vehicle_velocity.z)

    s_map = cal_s_map_fun(local_frenet_path_opt, origin_xy=(vehicle_loc.x, vehicle_loc.y))
    # 计算规划起点的s
    s0 = max(cur_path_s[0], 0)
    s = s0 + speed * ts
    # print("cur_path_s", cur_path_s)
    # print("predicted s", s)
    index = np.argmin(abs(np.array(cur_path_s) - s))
    l = cur_path_l[index]
    proj_x, proj_y, proj_theta, proj_kappa, pre_match_index = cal_proj_point_1(s, 0, local_frenet_path_opt, s_map)
    nor_v = np.array([-math.sin(proj_theta), math.cos(proj_theta)])  # 法向量
    pred_x, pred_y = np.array([proj_x, proj_y]) + l * nor_v

    return pred_x, pred_y


def cal_proj_point_1(s, pre_match_index, frenet_path_opt: list, s_map: list):
    # 确定s在s_map中的索引
    start_s_match_index = pre_match_index
    while s_map[start_s_match_index + 1] < s:  # 这里存在一点问题，如果动态规划采样点过长，会超出s_map的范围
        start_s_match_index += 1
    mp_x, mp_y, mp_theta, mp_kappa = frenet_path_opt[start_s_match_index]  # 取出投影点的路径信息
    ds = s - s_map[start_s_match_index]  # 计算规划起点的投影点和匹配点之间的位矢
    mp_tou_v = np.array([math.cos(mp_theta), math.sin(mp_theta)])  # 速度切线方向
    r_m = np.array([mp_x, mp_y])  # 匹配点位矢
    proj_x, proj_y = r_m + ds * mp_tou_v  # 近似投影点位置矢量
    proj_theta = mp_theta + mp_kappa * ds
    proj_kappa = mp_kappa
    res = (proj_x, proj_y, proj_theta, proj_kappa, start_s_match_index)
    return res


def cal_quintic_coefficient(start_l, start_dl, start_ddl, end_l, end_dl, end_ddl, start_s, end_s):
    A = np.array(
        [[1, start_s, pow(start_s, 2), pow(start_s, 3), pow(start_s, 4), pow(start_s, 5)],
         [0, 1, 2 * start_s, 3 * pow(start_s, 2), 4 * pow(start_s, 3), 5 * pow(start_s, 4)],
         [0, 0, 2, 6 * start_s, 12 * pow(start_s, 2), 20 * pow(start_s, 3)],
         [1, end_s, pow(end_s, 2), pow(end_s, 3), pow(end_s, 4), pow(end_s, 5)],
         [0, 1, 2 * end_s, 3 * pow(end_s, 2), 4 * pow(end_s, 3), 5 * pow(end_s, 4)],
         [0, 0, 2, 6 * end_s, 12 * pow(end_s, 2), 20 * pow(end_s, 3)]]
    )
    B = np.array([start_l, start_dl, start_ddl, end_l, end_dl, end_ddl])
    B = B.reshape((6, 1))
    coeffi = np.linalg.inv(A) @ B
    return list(coeffi.squeeze())


def Frenet2Cartesian(s_set, l_set, dl_set, ddl_set, frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, index2s):
    """
        Frenet 转 Cartesian
    """
    # 由于不知道有多少个(s,l)要转化成直角坐标，因此做缓冲
    # 输出初始化
    x_set = np.ones((600, 1)) * np.nan
    y_set = np.ones((600, 1)) * np.nan
    heading_set = np.ones((600, 1)) * np.nan
    kappa_set = np.ones((600, 1)) * np.nan

    for i in range(len(s_set)):
        if np.isnan(s_set[i]):
            break
        # 计算(s,l)在frenet坐标轴上的投影
        proj_x, proj_y, proj_heading, proj_kappa = CalcProjPoint(s_set[i], frenet_path_x, frenet_path_y,
                                                                 frenet_path_heading, frenet_path_kappa, index2s)
        nor = np.array([-np.sin(proj_heading), np.cos(proj_heading)])
        point = np.array([proj_x, proj_y]) + l_set[i] * nor
        x_set[i] = point[0]
        y_set[i] = point[1]
        heading_set[i] = proj_heading + np.arctan(dl_set[i] / (1 - proj_kappa * l_set[i]))
        # 近似认为 kappa' == 0, frenet转cartesian公式
        kappa_set[i] = ((ddl_set[i] + proj_kappa * dl_set[i] * np.tan(heading_set[i] - proj_heading)) *
                        (np.cos(heading_set[i] - proj_heading) ** 2) / (1 - proj_kappa * l_set[i]) + proj_kappa) * \
                       np.cos(heading_set[i] - proj_heading) / (1 - proj_kappa * l_set[i])

    return x_set, y_set, heading_set, kappa_set


def CalcProjPoint(s, frenet_path_x, frenet_path_y, frenet_path_heading, frenet_path_kappa, s_map):
    """
        该函数将计算在frenet坐标系下，点(s,l)在frenet坐标轴的投影的直角坐标(proj_x,proj_y,proj_heading,proj_kappa).T
        s_map从进程中获得
    """
    # 先找匹配点的编号
    match_index = 1
    while s_map[match_index] < s:
        match_index += 1
    match_point = np.array([frenet_path_x[match_index], frenet_path_y[match_index]])
    match_point_heading = frenet_path_heading[match_index]
    match_point_kappa = frenet_path_kappa[match_index]
    ds = s - s_map[match_index]
    match_tor = np.array([np.cos(match_point_heading), np.sin(match_point_heading)])
    proj_point = match_point + ds * match_tor
    proj_heading = match_point_heading + ds * match_point_kappa
    proj_kappa = match_point_kappa
    proj_x = proj_point[0]
    proj_y = proj_point[1]
    return proj_x, proj_y, proj_heading, proj_kappa


def trajectory_index2s(trajectory_x, trajectory_y):
    """
        该函数将计算以trajectory的s 与 x y 的对应关系，可以看作是trajectory index2s
        params: trajectory_x_init, trajectory_y_init
    """
    n = len(trajectory_x)
    path_index2s = np.zeros(n)
    s = 0
    tmp = 0
    for i in range(1, len(trajectory_x)):
        if np.isnan(trajectory_x[i]):
            tmp = i
            break
        s += np.sqrt((trajectory_x[i] - trajectory_x[i - 1]) ** 2 + (trajectory_y[i] - trajectory_y[i - 1]) ** 2)
        path_index2s[i] = s
    # 计算出trajectory的长度
    if tmp == n - 1:
        path_s_end = path_index2s[-1]
    else:
        # 因为循环的退出条件为isnan(trajectory_x(i)) 所以 i 所对应的数为 nan
        path_s_end = path_index2s[tmp - 1]

    return path_index2s


def cal_dy_obs_deri(l_set, vx_set, vy_set, proj_heading_set, proj_kappa_set):
    """
        该函数将计算frenet坐标系下动态障碍物的s_dot, l_dot, dl/ds
    """
    n = 128
    # 输出初始化
    s_dot_set = np.ones(n) * np.nan
    l_dot_set = np.ones(n) * np.nan
    dl_set = np.ones(n) * np.nan

    for i in range(len(l_set)):
        if np.isnan(l_set[i]):
            break
        v_h = np.array([vx_set[i], vy_set[i]])
        n_r = np.array([-np.sin(proj_heading_set[i]), np.cos(proj_heading_set[i])])
        t_r = np.array([np.cos(proj_heading_set[i]), np.sin(proj_heading_set[i])])
        l_dot_set[i] = np.dot(v_h, n_r)
        s_dot_set[i] = np.dot(v_h, t_r) / (1 - proj_kappa_set[i] * l_set[i])
        # 向量法做cartesian与frenet的转换要更简单，但是也有缺点，向量法必须依赖速度加速度
        # l' = l_dot/s_dot 但是如果s_dot = 0 此方法就失效了
        if abs(s_dot_set[i]) < 1e-6:
            dl_set[i] = 0
        else:
            dl_set[i] = l_dot_set[i] / s_dot_set[i]

    return s_dot_set, l_dot_set, dl_set



class RoadOption(Enum):
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANE_FOLLOW = 4
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6

class global_path_planner(object):
    def __init__(self, world_map, sampling_resolution):
        self._map = world_map  # type: carla.Map
        self._sampling_resolution = sampling_resolution
        self._topology = None
        self._graph = nx.DiGraph()  # type: nx.DiGraph
        self._id_map = None
        self._road_to_edge = None

        # initiate the planner
        self._build_topology()
        self._build_graph()

    def get_topology_and_graph_info(self):
        return self._topology, self._graph, self._id_map, self._road_to_edge

    def _build_topology(self):
        self._topology = []
        for seg in self._map.get_topology():
            w1 = seg[0]  # type: carla.Waypoint
            w2 = seg[1]  # type: carla.Waypoint
            new_seg = dict()
            new_seg["entry"] = w1
            new_seg["exit"] = w2
            new_seg["path"] = []
            # 按照采样分辨率将w1和w2之间的路径点采样出来
            w1_loc = w1.transform.location  # type: carla.Location
            if w1_loc.distance(w2.transform.location) > self._sampling_resolution:
                # 如果起始路点和结束路点之间存在其他路点，则根据采样分辨率将中间点全部存储在new_seg["path"]中
                new_waypoint = w1.next(self._sampling_resolution)[0]  # 这里从起始路点的下一个开始，
                while new_waypoint.transform.location.distance(w2.transform.location) > self._sampling_resolution:
                    # 结束路点不会记录到new_seg["path"]中
                    new_seg["path"].append(new_waypoint)
                    new_waypoint = new_waypoint.next(self._sampling_resolution)[0]
            else:  # 如果起始路点和结束路点之间的距离小于或等于采样分辨率，则仍然让new_seg["path"]保持空列表
                # new_seg["path"].append(w1.next(self._sampling_resolution)[0])
                pass
            self._topology.append(new_seg)

    def _build_graph(self):
        # self._graph = nx.DiGraph()  # it is initializes in the
        self._id_map = dict()  # 字典类型，建立节点id和位置的对应{(x, y, z): id}
        self._road_to_edge = dict()  # 字典类型，建立road_id,section_id,lane_id 和边的对应关系

        for seg in self._topology:
            entry_waypoint = seg["entry"]  # type: carla.Waypoint
            exit_waypoint = seg["exit"]  # type: carla.Waypoint
            path = seg["path"]  # 不包含端点
            intersection = entry_waypoint.is_intersection
            road_id, section_id, lane_id = entry_waypoint.road_id, entry_waypoint.section_id, entry_waypoint.lane_id
            entry_xyz = entry_waypoint.transform.location
            entry_xyz = (np.round(entry_xyz.x, 2), np.round(entry_xyz.y, 2), np.round(entry_xyz.z, 2))  # 对小数长度进行限制
            exit_xyz = exit_waypoint.transform.location
            exit_xyz = (np.round(exit_xyz.x, 2), np.round(exit_xyz.y, 2), np.round(exit_xyz.z, 2))
            for xyz in entry_xyz, exit_xyz:
                if xyz not in self._id_map:
                    New_ID = len(self._id_map)
                    self._id_map[xyz] = New_ID
                    # 将新的节点加入graph
                    self._graph.add_node(New_ID, vertex=xyz)

            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]

            if road_id not in self._road_to_edge:
                self._road_to_edge[road_id] = dict()
            if section_id not in self._road_to_edge[road_id]:
                self._road_to_edge[road_id][section_id] = dict()
            # 会有左右车道和多车道的情况 举例 13: {0: {-1: (34, 46), 1: (47, 31)}}，
            # 即id为13的道路，包含一个section,这个section是双向单车道
            self._road_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_forward_vector = entry_waypoint.transform.rotation.get_forward_vector()  # 这里是入口节点的方向信息
            exit_forward_vector = exit_waypoint.transform.rotation.get_forward_vector()  # 这里是出口节点的方向信息，用于车辆规划路径时的转向

            # 将新的边加入graph
            self._graph.add_edge(u_of_edge=n1, v_of_edge=n2,
                                 length=len(path) + 1, path=path,
                                 entry_waypoint=entry_waypoint, exit_waypoint=exit_waypoint,
                                 entry_vector=entry_forward_vector, exit_vector=exit_forward_vector,
                                 net_vector=Vector_fun(entry_waypoint.transform.location,
                                                                      exit_waypoint.transform.location),
                                 intersection=intersection, type=RoadOption.LANE_FOLLOW)

    def _find_location_edge(self, loc: carla.Location):
        nearest_wp = self._map.get_waypoint(loc)  # type: carla.Waypoint
        # 现在面临一个问题，对于两个路段相接处的节点，定位的是前一个路段还是后一个路段,在路径规划中二者本质上没有区别，但是自己没有搞明白这个方法的原理
        # 测试的结果是在交叉路口或者弯道情况下，返回的是后一个路段； 在直线道路中返回的是前一个路段
        edge = None
        try:
            # 用最近的路点所在的road_id,section_id和lane_id来定位其所在的边
            edge = self._road_to_edge[nearest_wp.road_id][nearest_wp.section_id][nearest_wp.lane_id]
        except KeyError:
            pass
        return edge

    def _route_search(self, origin, destination):
        start_edge = self._find_location_edge(origin)  # 获取起点所在边
        end_edge = self._find_location_edge(destination)  # 获取终点所在边
        route = self._A_star(start_edge[0], end_edge[0])
        if route is None:  # 如果不可达就报错
            raise nx.NetworkXNoPath(f"Node {start_edge[0]} not reachable from {end_edge[0]}")
        route.append(end_edge[1])  # 可达的话就将终点所在边的右端点加入路径
        return route

    def _A_star(self, n_begin, n_end):
        route = []
        open_set = dict()  # 字典， 记录每个节点的父节点和最短路径
        closed_set = dict()
        open_set[n_begin] = (0, -1)  # 每个节点对应一个元组，第一个元素是节点到起点的最短路径，第二个元素是父节点的id

        def cal_heuristic(n):
            # hypot返回原点到一点的多维欧几里得距离
            return math.hypot(self._graph.nodes[n]['vertex'][0] - self._graph.nodes[n_end]['vertex'][0],
                              self._graph.nodes[n]['vertex'][1] - self._graph.nodes[n_end]['vertex'][1],
                              self._graph.nodes[n]['vertex'][2] - self._graph.nodes[n_end]['vertex'][2])

        while 1:
            if len(open_set) == 0:  # 终点不可达
                return None
            # find the node with minimum distance between n_begin in open_set
            c_node = min(open_set, key=lambda n: open_set[n][0] + cal_heuristic(n))
            # print(c_node)
            if c_node == n_end:
                closed_set[c_node] = open_set[c_node]
                del open_set[c_node]  # 如果当前节点是终点，则把该节点从open_set中移除，加入到close_set.
                break
            for suc in self._graph.successors(c_node):  # 处理当前所有节点的后继
                new_cost = self._graph.get_edge_data(c_node, suc)["length"]  # 当前节点到后继节点的cost
                if suc in closed_set:  # 如果访问过就不再访问
                    continue
                elif suc in open_set:  # 如果在即将访问的集合中，判断是否需要更新路径
                    if open_set[c_node][0] + new_cost < open_set[suc][0]:
                        open_set[suc] = (open_set[c_node][0] + new_cost, c_node)
                else:  # 如果是新节点，直接加入open_set中
                    open_set[suc] = (open_set[c_node][0] + new_cost, c_node)
            closed_set[c_node] = open_set[c_node]
            del open_set[c_node]  # 遍历过该节点，则把该节点从open_set中移除，加入到close_set.

        route.append(n_end)
        while 1:
            if closed_set[route[-1]][1] != -1:
                route.append(closed_set[route[-1]][1])  # 通过不断回溯找到最短路径
            else:
                break
        return list(reversed(route))

    @staticmethod
    def _closest_index(current_waypoint, waypoint_list):
        """
        确定waypoint_list中距离当前路点最近的路点的索引值
        param:  current_waypoint:
                waypoint_list:
        return: 整数， 索引值
        """
        min_distance = float('inf')  # 初始情况下设置为最大值
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    def search_path_way(self, origin, destination):
        """
        得到完整的由waypoint构成的完整路径
        param:  origin: 起点，carla.Location类型
                destination: 终点
        return: list类型，元素是(carla.Waypoint类型, edge["type"]),这里多加了一个边的类型进行输出，
                 是为了后面的更全面考虑某些道路规定的跟车或者超车行为
        """
        route = self._route_search(origin, destination)  # 获取A*的初步规划结果->list
        origin_wp = self._map.get_waypoint(origin)  # type: carla.Waypoint
        destination_wp = self._map.get_waypoint(destination)  # type: carla.Waypoint
        path_way = []

        # 第一段路径
        edge = self._graph.get_edge_data(route[0], route[1])
        path = [edge["entry_waypoint"]] + edge["path"] + [edge["exit_waypoint"]]
        clos_index = self._closest_index(origin_wp, path)
        for wp in path[clos_index:]:
            path_way.append((wp, edge["type"]))

        # 中间路径
        if len(route) > 3:  # 先判断是否有中间路径
            for index in range(1, len(route) - 2):
                edge = self._graph.get_edge_data(route[index], route[index + 1])
                path = edge["path"] + [edge["exit_waypoint"]]  # 每一段路段的终点是下一个路段的起点，所以这里不加起点
                for wp in path:
                    path_way.append((wp, edge["type"]))

        # 最后一段路径
        edge = self._graph.get_edge_data(route[-2], route[-1])
        # print(edge)
        path = edge["path"] + [edge["exit_waypoint"]]
        clos_index = self._closest_index(destination_wp, path)
        if clos_index != 0:  # 判断终点是否是当前路段的起点，如果不是，将后续的路点加入path_way;
            for wp in path[:clos_index + 1]:
                path_way.append((wp, edge["type"]))
        else:  # 如果是，后面的路段终点则在上个路段已经添加进path_way中，这里不进行重复操作
            pass
        return path_way


'''
控制算法
'''


import cvxopt
import numpy as np
import math
import carla
# from planner.planning_utils import cal_heading_kappa
from collections import deque

print_flag = False
# print_flag = True

class Lateral_MPC_controller(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway_xy_theta_kappa):
        self._vehicle_state = None
        self._vehicle_para = vehicle_para
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self._vehicle_Vx = 0
        self._target_path = pathway_xy_theta_kappa
        self._N = 6  # 预测区间
        self._P = 2  # 控制区间
        self._n = 4  # 状态变量长度
        self.A = np.zeros(shape=(self._n, self._n), dtype="float64")
        self.B = np.zeros(shape=(self._n, 1), dtype="float64")
        self.C = np.zeros(shape=(self._n, 1), dtype="float64")
        self.A_bar = None  # 离散化的A,B,C矩阵
        self.B_bar = None
        self.C_bar = None
        self.k_r = None  # 曲率
        self.e_rr = None
        self.min_index = 0
        # debug 变量， 用于显示预测点和投影点的位置
        self.x_pre = 0
        self.y_pre = 0
        self.x_pro = 0
        self.y_pro = 0

    def cal_vehicle_info(self):
        """
        通过调用方法获取车辆的状态信息
        :return: None
        """
        vehicle_loc = self._vehicle.get_location()
        x, y = vehicle_loc.x, vehicle_loc.y
        fi = self._vehicle.get_transform().rotation.yaw * (math.pi / 180)  # 车身横摆角，车轴和x轴的夹角
        V = self._vehicle.get_velocity()  # 航向角是车速与x轴夹角
        V_length = math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)
        beta = (math.atan2(V.y, V.x) - fi)  # 质心侧偏角，车速和车轴之间的夹角
        # print("beta", beta, "theta", math.atan2(V.y, V.x), "fi", fi)
        Vy = V_length * math.sin(beta)  # 车速在车身坐标系下的分量
        """这里又一个非常奇怪的错误，当Vx=-0.0022左右时，
        求解出来的A_bar有些值特别大，进而导致C和H的部分之非常大，引起H的秩为1的情况，使得二次规划求解崩溃
        这里特殊处理一下，让Vx的绝对值不能小于0.005，刚好也能避免Vx为零时的错误，后面就不用担心为零的错误了"""
        if V_length * math.cos(beta) < 0:
            Vx = -max(abs(V_length * math.cos(beta)), 0.005)
        else:
            Vx = max(V_length * math.cos(beta), 0.005)
        # print("Vx", Vx, "Vy", Vy)
        fi_dao = self._vehicle.get_angular_velocity().z * (math.pi / 180)
        self._vehicle_state = (x, y, fi, Vy, fi_dao)
        self._vehicle_Vx = Vx

    def cal_A_B_C_fun(self):
        V_x = self._vehicle_Vx
        V_x = V_x
        (a, b, Cf, Cr, m, Iz) = self._vehicle_para

        self.A[0][1] = 1

        self.A[1][1] = (Cf + Cr) / (m * V_x)
        self.A[1][2] = -(Cf + Cr) / m
        self.A[1][3] = (a * Cf - b * Cr) / (m * V_x)

        self.A[2][3] = 1

        self.A[3][1] = (a * Cf - b * Cr) / (Iz * V_x)
        self.A[3][2] = -(a * Cf - b * Cr) / Iz
        self.A[3][3] = (a * a * Cf + b * b * Cr) / (Iz * V_x)

        self.B[1][0] = -Cf / m
        self.B[3][0] = -a * Cf / Iz

        self.C[1][0] = (a*Cf + b*Cr)/(m*V_x) - V_x
        self.C[3][0] = (a**2*Cf + b**2*Cr)/(Iz*V_x)

    def cal_coefficient_of_discretion_fun(self):
        """
        根据Q,R和A,B计算K, 通过迭代黎卡提方程求解， P = Q + A^PA - A^PB(R+B^PB)'B^PA.其中A^是求转置，A'是求逆
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param R: 控制代价的权重对应的对角矩阵1*1， R越大越平稳，变化越小
        :param A: cal_A_B_fun模块的输出4*4
        :param B:
        :return: K, np.array类型
        """
        ts = 0.1  # 连续lqr离散化的时间间隔
        # 连续lqr离散化的时候， 系数矩阵相应发生变化，采用的是双线性变换
        temp = np.linalg.inv(np.eye(4) - (ts * self.A) / 2)
        self.A_bar = temp @ (np.eye(4) + (ts * self.A) / 2)
        self.B_bar = temp @ self.B * ts
        self.C_bar = temp @ self.C * ts * self.k_r * self._vehicle_Vx
        # 这里将theta_r_dao=k_r * s_dao看做常数，无飘移假设下theta_r_dao = k_r * Vx,
        # 这里做过验证，在无飘移的条件下，s_dao=Vx,微小的漂移量二者也很接近
        # print("self.C_bar.shape", self.C_bar.shape)

    def cal_error_k_fun(self, ts=0.01):
        """
        计算预测点和规划点的误差
        :param ts: 控制周期
        :param target_path: 规划路径构成的矩阵x, y是直角坐标系下位置， theta是速度方向与x轴夹角，k是路径在该点的曲率
        [(x1, y1, theta_1, k1),
         (x4, y3, theta_2, k2),
         ...]
        :param cur_state: 车辆当前位置(x, y, fi, V_x, V_y, fi')
        x,y是车辆测量得到的实际位置
        fi是航向角，车轴和x轴的夹角
        V_x, V_y是速度在车轴方向的分量和垂直车轴方向的分量
        fi'是fi的导数
        :return: 当前位置和目标位置在frenet坐标系下的横向误差e_rr，以及投影点的曲率k
        """
        x, y, fi, V_y, fi_dao = self._vehicle_state
        V_x = self._vehicle_Vx
        # 预测模块
        x = x + V_x * ts * math.cos(fi) - V_y * ts * math.sin(fi)
        y = y + V_y * ts * math.cos(fi) + V_x * ts * math.sin(fi)
        fi = fi + fi_dao * ts
        # V_x, V_y, fi_dao认为在相邻的周期内是不变的
        self.x_pre = x
        self.y_pre = y
        # 1.确定匹配点
        path_length = len(self._target_path)
        min_d = 10000

        for i in range(self.min_index, min(self.min_index + 50, path_length)):  # 当控制器是根据全局路径进行控制的时候，
        # 为了缩短匹配点需要记住上一个匹配点位置，前期没有做局部规划，所以可以这样优化
        # for i in range(0, path_length):  # 后面我们的已经做了局部规划的时候，不能再用索引来记录上一个匹配点位置，
            # 局部路径规划本来长度就短，可以不记录也行；如果想优化，我们可以通过记录绝对时间来实现，用字典类型，每个路径点对应一个绝对时间信息
            d = (self._target_path[i][0] - x) ** 2 + (self._target_path[i][1] - y) ** 2
            if d < min_d:
                min_d = d
                self.min_index = i
        min_index = self.min_index
        # print("min_index:", self.min_index)
        # 2.计算车身坐标系下的轴向向量和法向量
        tor_v = np.array([math.cos(self._target_path[min_index][2]), math.sin(self._target_path[min_index][2])])
        n_v = np.array([-math.sin(self._target_path[min_index][2]), math.cos(self._target_path[min_index][2])])

        # 3.计算匹配点指向实际位置的向量
        d_v = np.array([x - self._target_path[min_index][0], y - self._target_path[min_index][1]])

        # 4.计算e_d, e_s
        e_d = np.dot(n_v, d_v)
        e_s = np.dot(tor_v, d_v)

        # 获取投影点坐标
        self.x_pro, self.y_pro = np.array(
            [self._target_path[min_index][0], self._target_path[min_index][1]]) + e_s * tor_v

        # 5.计算theta_r
        # 曲率的定义是K = delta_theta / delta_s 然后对delta_s求极限
        # 平均曲率就是K = delta_theta / delta_s
        # delta_theta 就是切线转角，delta_s是弧长，
        # 我们要假设匹配点和投影点的航向并不相同，但是近似认为两点的曲率是相同的，这样才可以计算delta_theta
        theta_r = self._target_path[min_index][2] + \
                  self._target_path[min_index][3] * e_s  # 认为投影点和匹配点航向不同，相差一个d_theta = k_r*d_s
        # theta_r = self._target_path[min_index][2]  # delta_theta本质上就是一个很小的值，
        # apollo方案，将其近似为零，认为投影点和匹配点的航向角相同，这样是有误差的，个人更偏向于用不为零的近似

        # 6.计算e_d的导数
        e_d_dao = V_y * math.cos(fi - theta_r) + V_x * math.sin(fi - theta_r)

        # 7.计算e_fi
        # e_fi = fi - theta_r
        e_fi = math.sin(fi - theta_r)  # 这里为了防止角度的多值性，用sin(e_fi)近似e_fi， 一般情况下e_fi就是一个小量，所以可以近似

        # 8.计算S的导数
        S_dao = (V_x * math.cos(fi - theta_r) - V_y * math.sin(fi - theta_r)) / (
                    1 - self._target_path[min_index][3] * e_d)

        # 9.计算e_fi的导数
        e_fi_dao = fi_dao - self._target_path[min_index][3] * S_dao

        # 10.计算投影点的曲率，近似等于匹配点的曲率
        self.k_r = self._target_path[min_index][3]
        self.e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)
        if print_flag:
            print("error-e_d-e_fi:", e_d, e_fi)

    def cal_control_para_fun(self, Q, R, F):
        """
        根据A_bac, B_bar, C_bar计算X_k
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param F: 终端误差的权重对应的对角矩阵4*4
        :param R: 应该是控制代价的权重对应的对角矩阵1*1，这里我传入的就是一个数值， R越大越平稳，变化越小
        :return: None
        """
        # 计算M，C，Cc
        M = np.zeros(shape=((self._N+1)*self._n, self._n))
        M[0:self._n, :] = np.eye(self._n)
        for i in range(1, self._N + 1):
            M[i*self._n:(i+1)*self._n, :] = self.A_bar @ M[(i-1)*self._n:i*self._n, :]

        C = np.zeros(shape=((self._N + 1) * self._n, self._N * self._P))
        C[self._n:2*self._n, 0:self._P] = self.B_bar  # 这里self.B_bar是4*1维度的， 由于控制区间是self_P，因此这里有个按列复制的过程
        for i in range(2, self._N + 1):
            C[i * self._n:(i + 1) * self._n, (i-1) * self._P:i * self._P] = self.B_bar
            for j in range(i-2, -1, -1):
                C[i*self._n:(i+1)*self._n, j*self._P:(j+1)*self._P] = \
                    self.A_bar @ C[i*self._n:(i+1)*self._n, (j+1)*self._P:(j+2)*self._P]

        Cc = np.zeros(shape=((self._N+1)*self._n, 1))
        for i in range(1, self._N+1):
            Cc[self._n*i:self._n*(i+1), 0:1] = self.A_bar @ Cc[self._n*(i-1):self._n*i, 0:1] + self.C_bar

        # 计算Q_bar, R_bar
        Q_bar = np.zeros(shape=((self._N+1)*self._n, (self._N+1)*self._n))
        for i in range(self._N):
            Q_bar[i*self._n:(i+1)*self._n, i*self._n:(i+1)*self._n] = Q
        Q_bar[self._N*self._n:, self._N*self._n:] = F

        R_bar = np.zeros(shape=(self._N*self._P, self._N*self._P))
        for i in range(self._N):
            R_bar[i*self._P:(i+1)*self._P, i*self._P:(i+1)*self._P] = np.eye(self._P)*R

        # 计算代价函数的系数矩阵 x'.H.x + 2E'x
        # G = M.T @ Q_bar @ M
        # print("V_x", self._vehicle_Vx)
        H = C.T @ Q_bar @ C + R_bar
        E = C.T @ Q_bar @ Cc + C.T @ Q_bar @ M @ (np.array(self.e_rr).reshape(self._n, 1))
        # 解决二次规划需要转化为标准形式0.5*x'.H.x + f'x
        H = 2 * H
        f = 2 * E
        # print("H.shape", H.shape, np.linalg.matrix_rank(np.matrix(H)))
        # print("f.shape", f.shape, np.linalg.matrix_rank(np.matrix(f)))
        # 约束项
        lb = np.ones(shape=(self._N*self._P, 1))*(-1)
        ub = np.ones(shape=(self._N*self._P, 1))
        G = np.concatenate((np.identity(self._N*self._P), -np.identity(self._N*self._P)))  # （4n, 2n）
        h = np.concatenate((ub, -lb))  # (4n, 1)
        # print("G.shape", G.shape, "The rank of G", np.linalg.matrix_rank(np.matrix(G)))
        # print("h.shape", h.shape, "The rank of h", np.linalg.matrix_rank(np.matrix(h)))
        cvxopt.solvers.options['show_progress'] = False  # 程序没有问题之后不再输出中间过程
        # 计算时要将输入转化为cvxopt.matrix
        # 该方法返回值是一个字典类型，包含了很多的参数，其中x关键字对应的是优化后的解
        res = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f),
                                G=cvxopt.matrix(G), h=cvxopt.matrix(h))
        return res['x'][0]

    def _control(self):
        """
        MPC 计算最终控制量
        :param K: LQR输出
        :param e_rr: 误差输出
        :param delta_f: 前馈输出
        :return: u最终控制量
        """
        b = 1
        Q = np.eye(4)
        Q[0][0] = 250
        Q[1][1] = 1
        Q[2][2] = 50
        Q[3][3] = 1
        F = np.eye(4)
        R = b
        self.cal_vehicle_info()
        self.cal_A_B_C_fun()
        self.cal_error_k_fun(ts=0.1)
        self.cal_coefficient_of_discretion_fun()

        current_steering = self.cal_control_para_fun(Q, R, F)
        # print("raw steering:", current_steering)
        return current_steering

class Lateral_LQR_controller(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway_xy_theta_kappa):
        self._vehicle_para = vehicle_para
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self._vehicle_state = None
        self._vehicle_Vx = 0
        self._target_path = pathway_xy_theta_kappa
        self.A = np.zeros(shape=(4, 4), dtype="float64")
        self.B = np.zeros(shape=(4, 1), dtype="float64")
        self.K = None
        self.k_r = None
        self.e_rr = None
        self.delta_f = None
        self.min_index = 0
        # debug 变量， 用于显示预测点和投影点的位置
        self.x_pre = 0
        self.y_pre = 0
        self.x_pro = 0
        self.y_pro = 0

    def xy_list_2_target_path(self, pathway_xy_list):
        """
        将由坐标构成的路径转化为（x, y, theta, k）的形式
        :param pathway_xy_list: 【x0, y0, (x1, y1), ...】
        :return: [(x0, y0, theta0, k0), ...]
        """
        theta_list, kappa_list = cal_heading_kappa(pathway_xy_list)
        # self._target_path = smooth_reference_line(xy_list_ori)  # 对生成的原始轨迹进行平滑,这里只是做了个实验
        for i in range(len(theta_list)):
            self._target_path.append((pathway_xy_list[i][0], pathway_xy_list[i][1], theta_list[i], kappa_list[i]))

    def cal_vehicle_info(self):
        """
        通过调用方法获取车辆的状态信息
        :return: None
        """
        vehicle_loc = self._vehicle.get_location()
        x, y = vehicle_loc.x, vehicle_loc.y
        fi = self._vehicle.get_transform().rotation.yaw*(math.pi/180)  # 车身横摆角，车轴和x轴的夹角
        V = self._vehicle.get_velocity()  # 航向角是车速与x轴夹角
        V_length = math.sqrt(V.x*V.x + V.y*V.y + V.z*V.z)
        beta = math.atan2(V.y, V.x) - fi  # 质心侧偏角，车速和车轴之间的夹角
        # print("beta", beta, "fi", fi)
        Vy = V_length*math.sin(beta)  # 车速在车身坐标系下的分量
        Vx = V_length*math.cos(beta)
        # print("Vx", Vx, "Vy", Vy)
        fi_dao = self._vehicle.get_angular_velocity().z*(math.pi/180)
        self._vehicle_state = (x, y, fi, Vy, fi_dao)
        self._vehicle_Vx = Vx

    def cal_A_B_fun(self):
        """
        根据整车参数vehicle_para和V_x,通过公式计算A,B; 具体公式间第八讲总结
              vehicle_para: vehicle_para = (a, b, Cf, Cr, m, Iz)
                            a，b是前后轮中心距离车质心的距离
                            CF, Cr是前后轮的侧偏刚度
                            m是车的质量
                            Iz是车的转动惯量

              V_x: V_x是车辆速度在车轴方向的分量
        :return: 矩阵A和B, np.array type
        A的维度4*4
        B的维度4*1
        """
        V_x = self._vehicle_Vx
        V_x = V_x + 0.0001  # 因为速度有可能为零，加个小量，避免后面的除法运算报错
        (a, b, Cf, Cr, m, Iz) = self._vehicle_para

        self.A[0][1] = 1

        self.A[1][1] = (Cf + Cr) / (m * V_x)
        self.A[1][2] = -(Cf + Cr) / m
        self.A[1][3] = (a * Cf - b * Cr) / (m * V_x)

        self.A[2][3] = 1

        self.A[3][1] = (a * Cf - b * Cr) / (Iz * V_x)
        self.A[3][2] = -(a * Cf - b * Cr) / Iz
        self.A[3][3] = (a * a * Cf + b * b * Cr) / (Iz * V_x)

        self.B[1][0] = -Cf / m
        self.B[3][0] = -a * Cf / Iz

    def LQR_fun(self, Q, R):
        """
        根据Q,R和A,B计算K, 通过迭代黎卡提方程求解， P = Q + A^PA - A^PB(R+B^PB)'B^PA.其中A^是求转置，A'是求逆
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param R: 控制代价的权重对应的对角矩阵4*4， R越大越平稳，变化越小
        :param A: cal_A_B_fun模块的输出4*4
        :param B:
        :return: K, np.array类型
        """
        P = Q
        P_pre = Q
        max_itr = 5000
        eps = 0.1
        ts = 0.1  # 连续lqr离散化的时间间隔
        # 连续lqr离散化的时候， 系数矩阵相应发生变化，采用的是双线性变换
        temp = np.linalg.inv(np.eye(4) - (ts*self.A)/2)
        A = temp @ (np.eye(4) + (ts*self.A)/2)
        B = temp @ self.B * ts
        i = 0
        AT = A.T  # 4*4
        BT = B.T  # 1*4
        for i in range(max_itr):
            P = AT @ P @ A - (AT @ P @ B) @ np.linalg.inv(R + BT @ P @ B) @ (BT @ P @ A) + Q  # 要不断迭代
            if abs(P - P_pre).max() < eps:
                break
            P_pre = P
        if print_flag:
            print("黎卡提方程迭代次数：", i)  # 输出迭代的次数

        self.K = np.linalg.inv(BT @ P @ B + R) @ (BT @ P @ A)

    def cal_error_k_fun(self, ts=0.01):
        """
        计算预测点和规划点的误差
        :param ts: 控制周期
        :param target_path: 规划路径构成的矩阵x, y是直角坐标系下位置， theta是速度方向与x轴夹角，k是路径在该点的曲率
        [(x1, y1, theta_1, k1),
         (x4, y3, theta_2, k2),
         ...]
        :param cur_state: 车辆当前位置(x, y, fi, V_x, V_y, fi')
        x,y是车辆测量得到的实际位置
        fi是航向角，车轴和x轴的夹角
        V_x, V_y是速度在车轴方向的分量和垂直车轴方向的分量
        fi'是fi的导数
        :return: 当前位置和目标位置在frenet坐标系下的横向误差e_rr，以及投影点的曲率k
        """
        x, y, fi, V_y, fi_dao = self._vehicle_state
        V_x = self._vehicle_Vx
        # 预测模块
        x = x + V_x * ts * math.cos(fi) - V_y * ts * math.sin(fi)
        y = y + V_y * ts * math.cos(fi) + V_x * ts * math.sin(fi)
        fi = fi + fi_dao * ts
        # V_x, V_y, fi_dao认为在相邻的周期内是不变的
        self.x_pre = x
        self.y_pre = y
        # 1.确定匹配点
        path_length = len(self._target_path)
        min_d = 10000

        # for i in range(self.min_index, min(self.min_index + 50, path_length)):  # 当控制器是根据全局路径进行控制的时候，
        # 为了缩短匹配点需要记住上一个匹配点位置，前期没有做局部规划，所以可以这样优化
        for i in range(0, path_length):  # 后面我们的已经做了局部规划的时候，不能再用索引来记录上一个匹配点位置，
            # 局部路径规划本来长度就短，可以不记录也行；如果想优化，我们可以通过记录绝对时间来实现，用字典类型，每个路径点对应一个绝对时间信息
            d = (self._target_path[i][0] - x) ** 2 + (self._target_path[i][1] - y) ** 2
            if d < min_d:
                min_d = d
                self.min_index = i
        min_index = self.min_index
        # print("min_index:", self.min_index)
        # 2.计算车身坐标系下的轴向向量和法向量
        tor_v = np.array([math.cos(self._target_path[min_index][2]), math.sin(self._target_path[min_index][2])])
        n_v = np.array([-math.sin(self._target_path[min_index][2]), math.cos(self._target_path[min_index][2])])

        # 3.计算匹配点指向实际位置的向量
        d_v = np.array([x - self._target_path[min_index][0], y - self._target_path[min_index][1]])

        # 4.计算e_d, e_s
        e_d = np.dot(n_v, d_v)
        e_s = np.dot(tor_v, d_v)

        # 获取投影点坐标
        self.x_pro, self.y_pro = np.array([self._target_path[min_index][0],
                                           self._target_path[min_index][1]]) + e_s * tor_v

        # 5.计算theta_r
        # 曲率的定义是K = delta_theta / delta_s 然后对delta_s求极限
        # 平均曲率就是K = delta_theta / delta_s
        # delta_theta 就是切线转角，delta_s是弧长，
        # 我们要假设匹配点和投影点的航向并不相同，但是近似认为两点的曲率是相同的，这样才可以计算delta_theta
        theta_r = self._target_path[min_index][2] + self._target_path[min_index][3] * e_s  # 认为投影点和匹配点航向不同，相差一个d_theta = k_r*d_s
        # theta_r = self._target_path[min_index][2]  # delta_theta本质上就是一个很小的值，
        # apollo方案，将其近似为零，认为投影点和匹配点的航向角相同，这样是有误差的，个人更偏向于用不为零的近似

        # 6.计算e_d的导数
        e_d_dao = V_y * math.cos(fi - theta_r) + V_x * math.sin(fi - theta_r)

        # 7.计算e_fi
        # e_fi = fi - theta_r
        e_fi = math.sin(fi - theta_r)  # 这里为了防止角度的多值性，用sin(e_fi)近似e_fi， 一般情况下e_fi就是一个小量，所以可以近似

        # 8.计算S的导数
        S_dao = (V_x * math.cos(fi - theta_r) - V_y * math.sin(fi - theta_r)) / (1 - self._target_path[min_index][3] * e_d)

        # 9.计算e_fi的导数
        e_fi_dao = fi_dao - self._target_path[min_index][3] * S_dao

        # 10.计算投影点的曲率，近似等于匹配点的曲率
        self.k_r = self._target_path[min_index][3]
        self.e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)
        if print_flag:
            print("error-e_d-e_fi:", e_d, e_fi)

    def forward_control_fun(self):
        """
        计算前馈控制量delta_f
        :param vehicle_para: vehicle_para = (a, b, Cf, Cr, m, Iz)
        :param K: LQR的输出结果
        :param k_r: 投影点曲率
        :param V_x: 速度在车轴方向的分量
        :return: 前馈空盒子量delta_f
        """
        a, b, Cf, Cr, m, Iz = self._vehicle_para
        # print(self.K.shape)
        K_3 = self.K[0][2]
        V_x = self._vehicle_Vx
        self.delta_f = self.k_r * (a + b - b * K_3 - (b / Cf + a * K_3 / Cr - a / Cr) * (m * V_x * V_x) / (a + b))
        self.delta_f = self.delta_f*np.pi/180  # 由于之前输入的误差形式弧度，所以这里也要将前馈量转化为弧度形式

    def _control(self):
        """
        LQR 计算最终控制量
        :param K: LQR输出
        :param e_rr: 误差输出
        :param delta_f: 前馈输出
        :return: u最终控制量
        """
        b = 1
        Q = np.eye(4)
        Q[0][0] = 200
        Q[1][1] = 1
        Q[2][2] = 50
        Q[3][3] = 1
        R = b
        self.cal_vehicle_info()
        self.cal_A_B_fun()
        self.LQR_fun(Q=Q, R=R)
        self.cal_error_k_fun(ts=0.1)
        self.forward_control_fun()
        # print("**********", self.K)
        # print("**********", self.delta_f)
        # print("**********", self.k_r)
        current_steering = -np.dot(self.K, np.array(self.e_rr)) + self.delta_f
        current_steering = current_steering[0]
        # print("raw steering:", current_steering)
        return current_steering


class Longitudinal_PID_controller(object):
    """
    PID 控制
    包括比例项， 积分项，微分项
    只有比例项会产生稳态误差，（稳态误差就是控制最终稳定在一个值但是和目标值有一定的差距）
    引入积分项可以消除稳态误差，但是会引起超调、震荡问题和积分饱和问题
    采用积分分离来克服系统超调和震荡
    """
    def __init__(self, ego_vehicle,  K_P=1.15, K_I=0, K_D=0, dt=0.01):
        """
        采用PID进行纵向控制
        :param ego_vehicle: 控制的车辆， 类型是carla.Vehicle
        :param K_P: 比例项系数
        :param K_I: 积分项系数
        :param K_D: 微分项系数
        :param dt: 控制间隔
        """
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D
        self.dt = dt
        self.target_speed = None
        self.error_buffer = deque(maxlen=60)  # 设置一个误差缓存区，用于积分项和差分项的计算
        self.error_threshold = 1  # 设定一个阈值，进行积分分离，标量单位是km/h,
        # 由于carla的最大throttle是1，因此误差如果大于1就让采取积分分离

    def PID_fun(self):
        """

        :return:
        """
        cur_velocity = self._vehicle.get_velocity()  # 调用carla API 获得的速度是个三维矢量 单位是m/s
        cur_speed = 3.6*math.sqrt(cur_velocity.x*cur_velocity.x
                                  + cur_velocity.y*cur_velocity.y
                                  + cur_velocity.z*cur_velocity.z)  # 转化为标量，单位是km/h

        error = self.target_speed - cur_speed  # 当前误差
        self.error_buffer.append(error)  # 将新的误差放入缓存区，如果缓存区满了，最左边的溢出，整体数据左移一位，新的数据加在最右边

        if len(self.error_buffer) >= 2:
            # 积分误差，为了解决稳态误差引入的积分项
            integral_error = sum(self.error_buffer) * self.dt
            # 微分误差，为了缓解超调
            differential_error = (self.error_buffer[-1] - self.error_buffer[-2]) / self.dt
        else:
            integral_error = 0.0
            differential_error = 0.0

        # 积分分离，当误差较大时，采取积分分离防止超调
        if print_flag:
            print("absolute speed error:", abs(error))

        if abs(error) > self.error_threshold:
            # 一旦出现误差大于阈值的情况，积分分离让积分项为0，清除误差缓存区，此时只有比例项发挥作用
            integral_error = 0.0
            self.error_buffer.clear()

        return self.K_P * error + self.K_I * integral_error + self.K_D * differential_error

    def PID_control(self, target_speed):

        self.target_speed = target_speed
        return self.PID_fun()


class Vehicle_control(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway, controller_type="MPC_controller"):
        self._vehicle = ego_vehicle
        self._max_throttle = 1
        self._max_brake = 1
        self._max_steer = 1
        self.min_steer = -1
        if controller_type == "MPC_controller":
            self.Lat_control = Lateral_MPC_controller(ego_vehicle, vehicle_para, pathway)
        elif controller_type == "LQR_controller":
            self.Lat_control = Lateral_LQR_controller(ego_vehicle, vehicle_para, pathway)

        self.Lon_control = Longitudinal_PID_controller(ego_vehicle)  # 这里不允许后续程序修改PID的参数，使用设定好的默认值

    def run_step(self, target_speed):

        control = carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False
        control.gear = 1
        current_steering = self.Lat_control._control()
        # 获取横向和纵向控制量
        current_acceleration = self.Lon_control.PID_control(target_speed)

        # 横向控制整定
        if current_steering >= 0:
            steering = min(self._max_steer, current_steering)
        else:
            steering = max(self.min_steer, current_steering)
        # steering = current_steering
        control.steer = steering

        # 纵向控制整定
        if current_acceleration >= 0:
            control.throttle = min(self._max_throttle, current_acceleration)
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = max(self._max_brake, current_acceleration)  # 没有反向加速，加速度为零时对应的是刹车制动

        V = self._vehicle.get_velocity()
        V_len = math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)
        if print_flag:
            print("current speed:", V_len, "m/s", "current steer control:", steering)
        return control


class Lateral_MPC__with_feedforward_controller(object):
    def __init__(self, ego_vehicle, vehicle_para, pathway_xy_theta_kappa):
        self._vehicle_state = None
        self._vehicle_para = vehicle_para
        self._vehicle = ego_vehicle  # type: carla.Vehicle
        self._vehicle_Vx = 0
        self._target_path = pathway_xy_theta_kappa
        self._N = 4  # 预测区间
        self._P = 2  # 控制区间
        self._n = 4  # 状态变量长度
        self.A = np.zeros(shape=(4, 4), dtype="float64")
        self.B = np.zeros(shape=(4, 1), dtype="float64")
        self.C = np.zeros(shape=(4, 1), dtype="float64")
        self.A_bar = None  # 离散化的A,B,C矩阵
        self.B_bar = None
        self.C_bar = None
        self.K = None  # 反馈增益
        self.k_r = None  # 曲率
        self.e_rr = None
        self.delta_f = None  # 前馈
        self.min_index = 0
        # debug 变量， 用于显示预测点和投影点的位置
        self.x_pre = 0
        self.y_pre = 0
        self.x_pro = 0
        self.y_pro = 0

        # 初始化
        self.cal_vehicle_info()
        self.cal_A_B_C_fun()

    def cal_vehicle_info(self):
        """
        acquire the states of ego-vehicle according to recall the internal methods
        通过调用方法获取车辆的状态信息
        :return: None
        """
        vehicle_loc = self._vehicle.get_location()
        x, y = vehicle_loc.x, vehicle_loc.y
        fi = self._vehicle.get_transform().rotation.yaw * (math.pi / 180)  # 车身横摆角，车轴和x轴的夹角
        V = self._vehicle.get_velocity()  # 航向角是车速与x轴夹角
        V_length = math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)
        beta = math.atan2(V.y, V.x) - fi  # 质心侧偏角，车速和车轴之间的夹角
        # print("beta", beta, "fi", fi)
        Vy = V_length * math.sin(beta)  # 车速在车身坐标系下的分量
        Vx = V_length * math.cos(beta)
        # print("Vx", Vx, "Vy", Vy)
        fi_dao = self._vehicle.get_angular_velocity().z * (math.pi / 180)
        self._vehicle_state = (x, y, fi, Vy, fi_dao)
        self._vehicle_Vx = Vx

    def cal_A_B_C_fun(self):
        """
        calculate the coefficient matrix
        根据整车参数vehicle_para和V_x,通过公式计算A,B; 具体公式间第八讲总结
              vehicle_para: vehicle_para = (a, b, Cf, Cr, m, Iz)
                            a，b是前后轮中心距离车质心的距离
                            CF, Cr是前后轮的侧偏刚度
                            m是车的质量
                            Iz是车的转动惯量

              V_x: V_x是车辆速度在车轴方向的分量
        """
        V_x = self._vehicle_Vx
        V_x = V_x + 0.0001  # 因为速度有可能为零，加个小量，避免后面的除法运算报错
        (a, b, Cf, Cr, m, Iz) = self._vehicle_para
        self.A[0][1] = 1

        self.A[1][1] = (Cf + Cr) / (m * V_x)
        self.A[1][2] = -(Cf + Cr) / m
        self.A[1][3] = (a * Cf - b * Cr) / (m * V_x)

        self.A[2][3] = 1

        self.A[3][1] = (a * Cf - b * Cr) / (Iz * V_x)
        self.A[3][2] = -(a * Cf - b * Cr) / Iz
        self.A[3][3] = (a * a * Cf + b * b * Cr) / (Iz * V_x)

        self.B[1][0] = -Cf / m
        self.B[3][0] = -a * Cf / Iz

        self.C[1][0] = (a*Cf + b*Cr)/(m*V_x) - V_x
        self.C[3][0] = (a**2*Cf + b**2*Cr)/(Iz*V_x)

    def cal_discretized_matrix(self):
        """
        calculate the discrete form of matrix A, B, C after the state equation is discretized.
        计算矩阵A、B、C的离散形式
        """
        ts = 0.1  # 连续lqr离散化的时间间隔
        # 连续lqr离散化的时候， 系数矩阵相应发生变化，采用的是双线性变换
        temp = np.linalg.inv(np.eye(4) - (ts * self.A) / 2)
        self.A_bar = temp @ (np.eye(4) + (ts * self.A) / 2)
        self.B_bar = temp @ self.B * ts
        self.C_bar = temp @ self.C * ts * self.k_r * self._vehicle_Vx  # 这里将theta_r_dao看做常数，无飘移假设下theta_r_dao = k_r * Vx
        # print("self.C_bar.shape", self.C_bar)

    def cal_error_k_fun(self, ts=0.01):
        """
        计算预测点和规划点的误差
        :param ts: 控制周期
        :param target_path: 规划路径构成的矩阵x, y是直角坐标系下位置， theta是速度方向与x轴夹角，k是路径在该点的曲率
        [(x1, y1, theta_1, k1),
         (x4, y3, theta_2, k2),
         ...]
        :param cur_state: 车辆当前位置(x, y, fi, V_x, V_y, fi')
        x,y是车辆测量得到的实际位置
        fi是航向角，车轴和x轴的夹角
        V_x, V_y是速度在车轴方向的分量和垂直车轴方向的分量
        fi'是fi的导数
        :return: 当前位置和目标位置在frenet坐标系下的横向误差e_rr，以及投影点的曲率k
        """
        x, y, fi, V_y, fi_dao = self._vehicle_state
        V_x = self._vehicle_Vx
        # 预测模块
        x = x + V_x * ts * math.cos(fi) - V_y * ts * math.sin(fi)
        y = y + V_y * ts * math.cos(fi) + V_x * ts * math.sin(fi)
        fi = fi + fi_dao * ts
        # V_x, V_y, fi_dao认为在相邻的周期内是不变的
        self.x_pre = x
        self.y_pre = y
        # 1.确定匹配点
        path_length = len(self._target_path)
        min_d = 10000

        # for i in range(self.min_index, min(self.min_index + 50, path_length)):  # 当控制器是根据全局路径进行控制的时候，
        # 为了缩短匹配点需要记住上一个匹配点位置，前期没有做局部规划，所以可以这样优化
        for i in range(0, path_length):  # 后面我们的已经做了局部规划的时候，不能再用索引来记录上一个匹配点位置，
            # 局部路径规划本来长度就短，可以不记录也行；如果想优化，我们可以通过记录绝对时间来实现，用字典类型，每个路径点对应一个绝对时间信息
            d = (self._target_path[i][0] - x) ** 2 + (self._target_path[i][1] - y) ** 2
            if d < min_d:
                min_d = d
                self.min_index = i
        min_index = self.min_index
        # print("min_index:", self.min_index)
        # 2.计算车身坐标系下的轴向向量和法向量
        tor_v = np.array([math.cos(self._target_path[min_index][2]), math.sin(self._target_path[min_index][2])])
        n_v = np.array([-math.sin(self._target_path[min_index][2]), math.cos(self._target_path[min_index][2])])

        # 3.计算匹配点指向实际位置的向量
        d_v = np.array([x - self._target_path[min_index][0], y - self._target_path[min_index][1]])

        # 4.计算e_d, e_s
        e_d = np.dot(n_v, d_v)
        e_s = np.dot(tor_v, d_v)

        # 获取投影点坐标
        self.x_pro, self.y_pro = np.array(
            [self._target_path[min_index][0], self._target_path[min_index][1]]) + e_s * tor_v

        # 5.计算theta_r
        # 曲率的定义是K = delta_theta / delta_s 然后对delta_s求极限
        # 平均曲率就是K = delta_theta / delta_s
        # delta_theta 就是切线转角，delta_s是弧长，
        # 我们要假设匹配点和投影点的航向并不相同，但是近似认为两点的曲率是相同的，这样才可以计算delta_theta
        theta_r = self._target_path[min_index][2] + self._target_path[min_index][
            3] * e_s  # 认为投影点和匹配点航向不同，相差一个d_theta = k_r*d_s
        # theta_r = self._target_path[min_index][2]  # delta_theta本质上就是一个很小的值，
        # apollo方案，将其近似为零，认为投影点和匹配点的航向角相同，这样是有误差的，个人更偏向于用不为零的近似

        # 6.计算e_d的导数
        e_d_dao = V_y * math.cos(fi - theta_r) + V_x * math.sin(fi - theta_r)

        # 7.计算e_fi
        e_fi = fi - theta_r
        # e_fi = math.sin(fi - theta_r)  # 这里为了防止角度的多值性，用sin(e_fi)近似e_fi， 一般情况下e_fi就是一个小量，所以可以近似

        # 8.计算S的导数
        S_dao = (V_x * math.cos(fi - theta_r) - V_y * math.sin(fi - theta_r)) / (
                    1 - self._target_path[min_index][3] * e_d)

        # 9.计算e_fi的导数
        e_fi_dao = fi_dao - self._target_path[min_index][3] * S_dao

        # 10.计算投影点的曲率，近似等于匹配点的曲率
        self.k_r = self._target_path[min_index][3]
        self.e_rr = (e_d, e_d_dao, e_fi, e_fi_dao)
        if print_flag:
            print("error-e_d-e_fi:", e_d, e_fi)

    def cal_control_para_fun(self, Q, R, F):
        """
        calculate the control variable(or signal)
        根据A_bac, B_bar, C_bar计算X_k
        :param Q: 是误差代价的权重对应的对角矩阵4*4,Q越大算法的性能越好，但是会牺牲算法稳定性导致最终控制量u很大
        :param F: 终端误差的权重对应的对角矩阵4*4
        :param R: 应该是控制代价的权重对应的对角矩阵1*1，这里我传入的就是一个数值， R越大越平稳，变化越小
        :return: None
        """
        # 计算M，C，Cc
        M = np.zeros(shape=((self._N+1)*self._n, self._n))
        C = np.zeros(shape=((self._N+1)*self._n, self._N*self._P))
        M[0:self._n, :] = np.eye(self._n)
        for i in range(1, self._N + 1):
            M[i*self._n:(i+1)*self._n, :] = self.A_bar @ M[(i-1)*self._n:i*self._n, :]

        C[self._n:2*self._n, 0:self._P] = self.B_bar  # 这里self.B_bar是4*1维度的， 由于控制区间是self_P，因此这里有个按复制的过程
        for i in range(2, self._N + 1):
            C[i * self._n:(i + 1) * self._n, (i-1) * self._P:i * self._P] = self.B_bar
            for j in range(i-2, -1, -1):
                C[i*self._n:(i+1)*self._n, j*self._P:(j+1)*self._P] = \
                    self.A_bar @ C[i*self._n:(i+1)*self._n, (j+1)*self._P:(j+2)*self._P]
        Cc = np.zeros(shape=((self._N+1)*self._n, 1))
        for i in range(1, self._N+1):
            Cc[self._n*i:self._n*(i+1), 0:1] = self.A_bar @ Cc[self._n*(i-1):self._n*i, 0:1] + self.C_bar

        # 计算Q_bar, R_bar
        Q_bar = np.zeros(shape=((self._N+1)*self._n, (self._N+1)*self._n))
        for i in range(self._N):
            Q_bar[i*self._n:(i+1)*self._n, i*self._n:(i+1)*self._n] = Q
        Q_bar[self._N*self._n:, self._N*self._n:] = F
        R_bar = np.zeros(shape=(self._N*self._P, self._N*self._P))
        for i in range(self._P):
            R_bar[i*self._P:(i+1)*self._P, i*self._P:(i+1)*self._P] = np.eye(self._P)*R

        # 计算代价函数的系数矩阵
        # G = M.T @ Q_bar @ M
        E = M.T @ Q_bar @ C
        H = C.T @ Q_bar @ C + R_bar
        # 解决二次规划需要转化为标准形式0.5*x'.H.x + f'x
        H = 2 * H
        f = 2 * E.T @ (np.array(self.e_rr).reshape(self._n, 1)) + 2 * C.T @ Q_bar.T @ Cc
        # print("H.shape", H.shape)
        # print("f.shape", f.shape)
        # 约束项
        lb = np.ones(shape=(self._N*self._P, 1))*(-1)
        ub = np.ones(shape=(self._N*self._P, 1))
        G = np.concatenate((np.identity(self._N*self._P), -np.identity(self._N*self._P)))  # （4n, 2n）
        h = np.concatenate((ub, -lb))  # (4n, 1)
        # print("G.shape", G.shape)
        # print("h.shape", h.shape)
        cvxopt.solvers.options['show_progress'] = False  # 程序没有问题之后不再输出中间过程
        # 计算时要将输入转化为cvxopt.matrix
        # 该方法返回值是一个字典类型，包含了很多的参数，其中x关键字对应的是优化后的解
        res = cvxopt.solvers.qp(cvxopt.matrix(H), cvxopt.matrix(f), G=cvxopt.matrix(G), h=cvxopt.matrix(h))
        return res['x'][0]

    def MPC_control(self):
        """
        计算最终控制量
        :param K: LQR输出
        :param e_rr: 误差输出
        :param delta_f: 前馈输出
        :return: u最终控制量
        """
        b = 1
        Q = np.eye(4)
        Q[0][0] = 200
        Q[1][1] = 1
        Q[2][2] = 1
        Q[3][3] = 1
        F = 10 * np.eye(4)
        R = b
        self.cal_vehicle_info()
        self.cal_A_B_C_fun()
        self.cal_error_k_fun(ts=0.1)
        self.cal_discretized_matrix()
        # print("**********", self.K)
        # print("**********", self.delta_f)
        # print("**********", self.k_r)
        current_steering = self.cal_control_para_fun(Q, R, F)
        # current_steering = current_steering[0]
        # print("raw steering:", current_steering)
        return current_steering

'''
局部规划路径处理
'''

from agents.navigation.global_route_planner import GlobalRoutePlanner

class VehicleAgent:
    def __init__(self, name, vehicle, world, map, start_location, end_location, vehicle_para, initial_lane, controller_type="MPC_controller", color=carla.Color(255, 0, 0)):
        """
        初始化单个车辆代理对象
        :param name: 车辆名称（字符串，用于标识）
        :param vehicle: CARLA 仿真中的车辆对象
        :param world: CARLA 世界对象
        :param map: CARLA 地图对象
        :param start_location: 全局路径的起点 (carla.Location)
        :param end_location: 全局路径的终点 (carla.Location)
        :param vehicle_para: 车辆参数
        :param initial_lane: 初始车道编号 (0=左车道, 1=中车道, 2=右车道)
        :param controller_type: 控制器类型
        """
        self.name = name
        self.vehicle = vehicle
        self.id = vehicle.id  # 使用 Carla Actor 的 ID 属性作为 VehicleAgent 的 ID
        self.world = world
        self.map = map
        self.start_location = start_location
        self.end_location = end_location
        self.vehicle_para = vehicle_para
        self.controller_type = controller_type
        self.color = color  # 车辆轨迹的颜色
        self.trajectory_points = []  # 存储车辆历史轨迹点


        # 设置当前车道为初始车道
        self.current_lane = initial_lane

        # 计算全局路径
        self.global_path = self.calculate_global_path(start_location, end_location)
        self.current_path = self.global_path  # 当前路径
        self.controller = Vehicle_control(vehicle, vehicle_para, self.global_path, controller_type)

        self.is_in_lane_change = False  # 标记是否正在变道
        self.global_index = 0  # 当前路径中的索引

        # 绘制全局路径的散点
        # draw_global_path(world, self.global_path, color=(color.r, color.g, color.b))

    def get_transform(self):
        """
        获取车辆的 transform 信息（位置和旋转）
        """
        return self.vehicle.get_transform()

    def get_location(self):
        """
        获取车辆的当前位置
        """
        return self.vehicle.get_location()

    def set_autopilot(self, enable=True):
        """
        启用或禁用车辆的自动驾驶模式
        """
        self.vehicle.set_autopilot(enable)

    def draw_trajectory(self):
        """绘制车辆的历史轨迹"""
        update_vehicle_trajectory_with_points(self.vehicle, self.world, self.trajectory_points, color=self.color)

    def calculate_global_path(self, start_location, end_location):
        """计算全局路径"""
        resolution = 2
        planner = GlobalRoutePlanner(self.map, resolution)
        pathway = planner.trace_route(
            self.map.get_waypoint(start_location, project_to_road=True).transform.location,
            self.map.get_waypoint(end_location, project_to_road=True).transform.location
        )
        global_path = waypoint_list_2_target_path(pathway)
        # print(f"{self.name} 全局路径计算完成")
        return global_path

    def find_nearest_path_point(self, vehicle_x, vehicle_y):
        """找到车辆最近的路径点索引"""
        distances = [math.sqrt((vehicle_x - point[0]) ** 2 + (vehicle_y - point[1]) ** 2) for point in self.current_path]
        return np.argmin(distances)

    def extract_local_path(self, change_lane_length=20):
        """提取局部路径"""
        end_index = min(self.global_index + change_lane_length, len(self.current_path))
        return self.current_path[self.global_index:end_index]

    def get_vehicle(self):
        return self.vehicle

    def plan_lane_change(self, decision, s_end=20, lane_width=3.5):
        """
        规划变道轨迹
        :param decision: 决策 (time, action)，例如 (3, 0)。
                         action: 0=向左变道, 1=沿当前车道行驶, 2=向右变道
        :param s_end: Frenet 坐标系中的终点 s 坐标
        :param lane_width: 每条车道的宽度
        """
        # 基于时间的决策
        # time, action = decision
        # 不基于时间的决策
        action = decision

        # 解释决策含义
        # print(f"{self.name} 收到决策: {decision}")
        if action == 1:  # 保持当前车道
            #print(f"{self.name} 保持在当前车道 {self.current_lane}，无需变道。")
            return

        if self.is_in_lane_change:
            #print(f"{self.name} 正在变道中，无法连续变道。")
            return

        # 根据 action 计算目标车道
        if action == 0:  # 向左变道
            target_lane = max(0, self.current_lane - 1)  # 避免越界
            d_offset = -lane_width
        elif action == 2:  # 向右变道
            target_lane = min(2, self.current_lane + 1)  # 避免越界
            d_offset = lane_width
        else:
            # print(f"{self.name} 收到未知决策: {action}，忽略。")
            return

        # 提取局部路径
        local_path = self.extract_local_path(20)
        # print(f"{self.name} 提取局部路径: {local_path}")

        # 转换为 Frenet 坐标
        frenet_path = cartesian_to_frenet(local_path)
        # print(f"{self.name} Frenet 路径: {frenet_path}")

        # 设置变道条件
        s_ego = 0  # 自车初始 s 坐标
        # d_ego = lane_width * self.current_lane  # 当前车道的横向位置
        # d_end = lane_width * target_lane  # 目标车道的横向位置
        d_ego = 0
        d_end = d_offset

        # 生成变道轨迹
        trajectory = lane_change_trajectory(frenet_path, s_ego, s_end, d_ego, d_end)
        #print(f"{self.name} 生成变道轨迹: {trajectory}")

        # 转换回笛卡尔坐标并更新路径
        cartesian_path = frenet_to_cartesian(local_path, trajectory)
        #print(f"{self.name} 笛卡尔变道轨迹: {trajectory}")
        transition_path = cartesian_to_path(cartesian_path)
        #print(f"{self.name} Carla 路径变道轨迹: {transition_path}")
        self.current_path = transition_path
        self.global_index = 0  # 重置索引到变道轨迹的起点
        self.is_in_lane_change = True  # 标记正在变道

        # 更新控制器
        self.controller = Vehicle_control(self.vehicle, self.vehicle_para, self.current_path, self.controller_type)
        draw_debug_path(self.world, self.current_path, color=(0, 255, 0))

        # 更新当前车道
        self.current_lane = target_lane
        #print(f"{self.name} 变道完成，当前车道为 {self.current_lane}")

    def restore_global_path(self):
        """变道完成后，重新计算全局路径"""
        # 获取变道轨迹的终点作为新的起点
        last_point = self.current_path[-1]
        new_start_location = carla.Location(x=last_point[0], y=last_point[1], z=2.0)

        # 重新计算全局路径
        self.global_path = self.calculate_global_path(new_start_location, self.end_location)
        self.current_path = self.global_path  # 更新当前路径
        self.global_index = 0  # 重置索引到全局路径起点
        self.controller = Vehicle_control(self.vehicle, self.vehicle_para, self.global_path, self.controller_type)
        # print(f"{self.name} 恢复全局路径")

    # def control_step(self, target_speed=15 * 3.6):
    #     """控制车辆沿当前路径行驶"""
    #     control = self.controller.run_step(target_speed)
    #     self.vehicle.apply_control(control)

    def control_step(self, vertical_behavior, front_vehicle_speed, max_speed=13.8 * 3.6, lane_change_speed=11.1 * 3.6):
        current_speed = self.vehicle.get_velocity().length()  # 当前速度
        new_speed, target_speed = speed_planning(
            vertical_behavior,
            current_speed * 3.6,
            max_speed,
            self.is_in_lane_change,
            a_max=3,  # 最大加速度
            a_min=-4.5,  # 最大减速度
            lane_change_speed=lane_change_speed,
            front_vehicle_speed=front_vehicle_speed
        )
        control = self.controller.run_step(new_speed)
        # control = self.controller.run_step(30)
        self.vehicle.apply_control(control)

    def update(self):
        """更新车辆状态"""
        vehicle_loc = self.vehicle.get_transform().location
        self.global_index = self.find_nearest_path_point(vehicle_loc.x, vehicle_loc.y)
        #print(f"{self.name} 当前路径索引: {self.global_index}")

        # 如果变道完成，恢复全局路径
        if self.is_in_lane_change and self.global_index >= len(self.current_path) - 1:
            #print(f"{self.name} 变道完成")
            self.is_in_lane_change = False  # 重置变道状态
            self.restore_global_path()
    # def update(self, behavior_data_json_import, max_speed, lane_change_speed, a_max=3, a_min=-4.5):
    #     """
    #     更新车辆状态并执行决策。
    #     :param behavior_data_json_import: 实时行为数据
    #     :param max_speed: 道路最大速度
    #     :param lane_change_speed: 变道时的速度
    #     :param a_max: 最大加速度
    #     :param a_min: 最大减速度
    #     """
    #     # 提取车辆的实时行为数据
    #     behavior, vertical_behavior, front_vehicle_speed = extract_behavior_decisions(behavior_data_json_import,
    #                                                                                   self.name)
    #
    #     # 1. 横向行为：根据横向决策（变道逻辑）
    #     if behavior is not None and behavior != 1:  # 如果有变道需求
    #         self.plan_lane_change((0, behavior))  # `0` 表示即时决策
    #
    #     # 2. 纵向行为：根据速度规划动态调整速度
    #     current_speed = self.vehicle.get_velocity().length()  # 当前速度
    #     new_speed, target_speed = speed_planning(
    #         vertical_behavior, current_speed, max_speed, self.is_in_lane_change, a_max, a_min, lane_change_speed,
    #         front_vehicle_speed
    #     )
    #
    #     # 打印调试信息
    #     print(f"{self.name} 当前行为: 横向决策={behavior}, 纵向决策={vertical_behavior}, 前车速度={front_vehicle_speed}")
    #     print(f"{self.name} 当前速度: {current_speed:.2f}, 目标速度: {target_speed:.2f}, 更新速度: {new_speed:.2f}")
    #
    #     # 控制器更新目标速度
    #     self.controller.update_target_speed(new_speed)
    #
    #     # 更新车辆的状态
    #     vehicle_loc = self.vehicle.get_transform().location
    #     self.global_index = self.find_nearest_path_point(vehicle_loc.x, vehicle_loc.y)
    #
    #     # 如果变道完成，恢复全局路径
    #     if self.is_in_lane_change and self.global_index >= len(self.current_path) - 1:
    #         print(f"{self.name} 变道完成")
    #         self.is_in_lane_change = False  # 重置变道状态
    #         self.restore_global_path()

def update_vehicle_trajectory_with_points(vehicle, world, trajectory_points, color=carla.Color(255, 0, 0), point_size=0.1, point_life_time=10):
    """
    更新车辆行驶轨迹并使用散点绘制
    :param vehicle: carla.Vehicle 对象
    :param world: carla.World 对象
    :param trajectory_points: 历史轨迹点列表，用于存储车辆的轨迹
    :param color: 散点的颜色
    :param point_size: 散点的大小
    :param point_life_time: 散点的可视化持续时间
    """
    # 获取车辆当前位置
    current_location = vehicle.get_location()

    # 绘制当前位置为一个散点
    world.debug.draw_point(
        current_location,
        size=point_size,
        color=color,
        life_time=point_life_time
    )

    # 将当前点添加到历史点列表（用于其他用途，如分析轨迹）
    trajectory_points.append(current_location)

# def extract_behavior_decisions(behavior_data_json_import, vehicle_name):
#     """
#     从行为数据中提取指定车辆的决策信息。
#     :param behavior_data_json_import: 行为数据字典
#     :param vehicle_name: 当前车辆的 ID，例如 "QD1E003P"
#     :return: 提取的横向行为、纵向行为、前车速度
#     """
#     vehicle_behavior_data = behavior_data_json_import.get("result", {}).get("vehicle_behavior_priority", {}).get(vehicle_name, {})
#     behavior = vehicle_behavior_data.get("behavior", None)  # 横向决策
#     vertical_behavior = vehicle_behavior_data.get("vertical_behavior", None)  # 纵向决策
#     front_vehicle_speed = vehicle_behavior_data.get("front_vehicle_speed", None)  # 前车速度
#     return behavior, vertical_behavior, front_vehicle_speed

def get_realtime_decision(behavior_data_json, vehicle_id):
    vehicle_behavior_data = behavior_data_json.get("result", {}).get("vehicle_behavior_priority", {}).get(vehicle_id, {})
    behavior = vehicle_behavior_data.get("behavior", None)  # 横向决策
    vertical_behavior = vehicle_behavior_data.get("vertical_behavior", None)  # 纵向决策
    front_vehicle_speed = vehicle_behavior_data.get("front_vehicle_speed", None)  # 前车速度
    # behavior = vehicle_behavior_data.get("QD1E003P", {}).get("behavior", None)
    # vertical_behavior = vehicle_behavior_data.get("QD1E003P", {}).get("vertical_behavior", None)
    # front_vehicle_speed = vehicle_behavior_data.get("QD1E003P", {}).get("front_vehicle_speed", None)
    return behavior, vertical_behavior, front_vehicle_speed






def waypoint_list_2_target_path(pathway):
    """
    将由路点构成的路径转化为(x, y, theta, k)的形式
    param: pathway: [waypoint0, waypoint1, ...]
    return: [(x0, y0, theta0, k0), ...]
    """
    target_path = []
    w = None  # type: carla.Waypoint
    xy_list_ori = []
    for w in pathway:
        x = w[0].transform.location.x
        y = w[0].transform.location.y
        xy_list_ori.append((x, y))

    theta_list, kappa_list = cal_heading_kappa(xy_list_ori)  # 包含frenet曲线上每一点的坐标[(x0,y0), (x1, y1), ...]
    # self._target_path = smooth_reference_line(xy_list_ori)  # 对生成的原始轨迹进行平滑,这里只是做了个实验
    for i in range(len(theta_list)):
        target_path.append((xy_list_ori[i][0], xy_list_ori[i][1], theta_list[i], kappa_list[i]))
    return target_path


def cal_heading_kappa(frenet_path_xy_list: list):
    dx_ = []
    dy_ = []
    for i in range(len(frenet_path_xy_list) - 1):
        dx_.append(frenet_path_xy_list[i + 1][0] - frenet_path_xy_list[i][0])
        dy_.append(frenet_path_xy_list[i + 1][1] - frenet_path_xy_list[i][1])
    # 计算theta,切线方向角
    # 由于n个点差分得到的只有n-1个差分结果，所以要在首尾添加重复单元来近似求每个节点的dx,dy
    dx_pre = [dx_[0]] + dx_  # 向前补dx_的第一位
    dx_aft = dx_ + [dx_[-1]]  # 向后补dx_的最后一位
    dx = (np.array(dx_pre) + np.array(dx_aft)) / 2

    dy_pre = [dy_[0]] + dy_
    dy_aft = dy_ + [dy_[-1]]
    dy = (np.array(dy_pre) + np.array(dy_aft)) / 2
    theta = np.arctan2(dy, dx)  # np.arctan2会将角度限制在（-pi, pi）之间
    # 计算曲率
    d_theta_ = np.diff(theta)  # 差分计算
    d_theta_pre = np.insert(d_theta_, 0, d_theta_[0])
    d_theta_aft = np.insert(d_theta_, -1, d_theta_[-1])
    d_theta = np.sin((d_theta_pre + d_theta_aft) / 2)  # 认为d_theta是个小量，用sin(d_theta)代替d_theta,避免多值性
    ds = np.sqrt(dx ** 2 + dy ** 2)
    k = d_theta / ds

    return list(theta), list(k)

def generate_fitted_path_mid_section(points_x_1, points_y_1, points_x_0, points_y_0, num_points=100):
    """
    根据输入点顺序拟合五次多项式路径，并保留中间部分轨迹。

    参数:
    - points_x_1: 起点的 x 坐标列表。
    - points_y_1: 起点的 y 坐标列表。
    - points_x_0: 终点的 x 坐标列表。
    - points_y_0: 终点的 y 坐标列表。
    - num_points: 生成路径点的数量。

    返回:
    - path_points: 保留中间部分的路径点列表 [(x1, y1), (x2, y2), ...]。
    """
    # 合并起点和终点的坐标（保持输入顺序）
    x_points = np.array(points_x_1 + points_x_0)
    y_points = np.array(points_y_1 + points_y_0)

    # 五次多项式拟合
    coefficients = np.polyfit(x_points, y_points, 5)

    # 生成路径点的 x 坐标（按输入点顺序生成）
    path_x = np.linspace(x_points[0], x_points[-1], num_points)

    # 计算路径点的 y 坐标
    path_y = np.polyval(coefficients, path_x)

    # 保留中间部分（去掉前 15 个点和后 15 个点，只保留中间 70 个点）
    mid_start = 15
    mid_end = num_points - 15
    path_x_mid = path_x[mid_start:mid_end]
    path_y_mid = path_y[mid_start:mid_end]

    # 返回路径点
    path_points = [(x, y) for x, y in zip(path_x_mid, path_y_mid)]
    return path_points


def trajectory_planning(global_index, current_path, target_path):
    """
    基于当前索引进行变道规划，并返回变道轨迹。
    """
    if global_index + 2 < len(target_path):
        index10 = global_index
        index11 = global_index + 1
        index12 = global_index + 2

        index00 = global_index + 20
        if index00 + 2 >= len(target_path):
            return current_path, None

        index01 = index00 + 1
        index02 = index00 + 2

        # 当前路径起点
        points_x_1 = [current_path[index10][0], current_path[index11][0], current_path[index12][0]]
        points_y_1 = [current_path[index10][1], current_path[index11][1], current_path[index12][1]]

        # 目标路径终点
        points_x_0 = [target_path[index00][0], target_path[index01][0], target_path[index02][0]]
        points_y_0 = [target_path[index00][1], target_path[index01][1], target_path[index02][1]]

        # 打印路径点验证是否正确
        # print(f"起点路径点: {points_x_1}, {points_y_1}")
        # print(f"终点路径点: {points_x_0}, {points_y_0}")

        # 拟合平滑过渡轨迹
        path_points = generate_fitted_path_mid_section(points_x_1, points_y_1, points_x_0, points_y_0, num_points=100)

        return path_points, target_path  # 返回变道轨迹和目标路径

    return current_path, None


def draw_debug_path(world, path, color=(0, 255, 0), z_offset=0.5):
    """
    在 Carla 中绘制路径。
    参数:
    - world: Carla 世界对象。
    - path: 路径点列表 [(x, y, theta, kappa), ...]。
    - color: 路径点的颜色 (R, G, B)。
    - z_offset: 高度偏移量。
    """
    for point in path:
        world.debug.draw_point(
            carla.Location(x=point[0], y=point[1], z=z_offset),
            size=0.1,
            color=carla.Color(r=color[0], g=color[1], b=color[2]),
            life_time=10
        )

def find_nearest_path_point(vehicle_x, vehicle_y, path):
    """找到距离车辆最近的路径点"""
    # 解包路径点的 x, y 值，忽略其他字段
    distances = [math.sqrt((vehicle_x - point[0]) ** 2 + (vehicle_y - point[1]) ** 2) for point in path]
    return np.argmin(distances)

def cal_heading_kappa(frenet_path_xy_list: list):
    """
    计算frenet曲线上每个点的切向角theta（与直角坐标轴之间的角度）和曲率kappa
    :param frenet_path_xy_list: 包含frenet曲线上每一点的坐标[(x0,y0), (x1, y1), ...]
    :return: theta列表和kappa列表
    """
    dx_ = []
    dy_ = []
    for i in range(len(frenet_path_xy_list) - 1):
        dx_.append(frenet_path_xy_list[i + 1][0] - frenet_path_xy_list[i][0])
        dy_.append(frenet_path_xy_list[i + 1][1] - frenet_path_xy_list[i][1])

    # 向前和向后扩展dx和dy以平滑
    dx_pre = [dx_[0]] + dx_
    dx_aft = dx_ + [dx_[-1]]
    dx = (np.array(dx_pre) + np.array(dx_aft)) / 2

    dy_pre = [dy_[0]] + dy_
    dy_aft = dy_ + [dy_[-1]]
    dy = (np.array(dy_pre) + np.array(dy_aft)) / 2

    # 计算切向角theta
    theta = np.arctan2(dy, dx)

    # 计算theta的差分，用于计算曲率
    d_theta_ = np.diff(theta)
    d_theta_pre = np.insert(d_theta_, 0, d_theta_[0])
    d_theta_aft = np.insert(d_theta_, len(d_theta_), d_theta_[-1])
    d_theta = np.sin((d_theta_pre + d_theta_aft) / 2)

    # 计算弧长增量ds
    ds = np.sqrt(dx ** 2 + dy ** 2)

    # 计算曲率kappa
    kappa = d_theta / ds

    return list(np.round(theta, 7)), list(np.round(kappa, 7))

def xy_list_2_target_path(pathway_xy_list: list):
    """
    将由坐标构成的路径转化为（x, y, theta, kappa）的形式
    :param pathway_xy_list: [(x0, y0), (x1, y1), ...]
    :return: [(x0, y0, theta0, kappa0), ...]
    """
    theta_list, kappa_list = cal_heading_kappa(pathway_xy_list)
    target_path = []
    for i in range(len(pathway_xy_list)):
        target_path.append((pathway_xy_list[i][0], pathway_xy_list[i][1], theta_list[i], kappa_list[i]))
    return target_path

# Helper function to process behavior data
def generate_behavior_data(behavior_data_json_import):
    """
    从传入的行为决策数据中提取并处理信息，并返回指定格式的行为数据 JSON。

    参数:
    - behavior_data_json_import: dict，外部程序传入的行为决策数据。

    返回:
    - dict，处理后的行为数据 JSON 格式
    """

    # 提取时间戳
    timestamp = behavior_data_json_import.get("timestamp", datetime.now().timestamp())

    # 提取车辆信息
    vehicles_info = []
    for vehicle in behavior_data_json_import.get("result", {}).get("topological_vehicles", []):
        vehicle_data = {
            "uuid": vehicle.get("uuid"),
            "vehicleId": vehicle.get("vehicleId"),
            "latitude": vehicle.get("lat"),
            "longitude": vehicle.get("lon"),
            "speed": vehicle.get("speed"),
            "length": vehicle.get("length"),
            "width": vehicle.get("width"),
        }
        vehicles_info.append(vehicle_data)

    # 提取行为优先级信息
    behavior_priority = behavior_data_json_import.get("result", {}).get("vehicle_behavior_priority", {})

    # 构造处理后的数据
    processed_data = {
        "timestamp": timestamp,
        "vehicles": vehicles_info,
        "vehicle_behavior_priority": behavior_priority,
    }

    return processed_data

from multiprocessing import Process, Pipe
import random

def behavior_data_sender(conn):
    """
    向管道发送模拟的行为决策信息。
    """
    while True:
        # 模拟的行为决策信息
        # random_behavior = random.choice([0, 1, 2])  # 随机选择 0, 1 或 2
        random_behavior = random.choices([0, 1, 2], weights=[1, 3, 1])[0]  # 根据权重选择 0, 1 或 2
        behavior_data_json_import = {
            "timestamp": datetime.now().timestamp(),
            "result": {
                "coop_set": ["QD1E003P"],
                "topological_vehicles": [
                    {
                        "crossId": None,
                        "heading": 233.8125,
                        "height": None,
                        "lat": 29.5223185,
                        "length": None,
                        "lon": 106.3227816,
                        "roadSegId": None,
                        "speed": 1.2,
                        "srcType": 1,
                        "type": 2,
                        "uuid": "1062dbd7883581d48dafb1f0",
                        "vehicleId": "QD1E003P",
                        "width": None
                    }
                ],
                "vehicle_behavior_priority": {
                    "QD1E003P": {
                        "behavior": random_behavior,  # 改变行为（0: 左车道, 1: 保持, 2: 右车道）
                        "priority": 1
                    }
                }
            }
        }

        # 模拟 ego_data
        ego_data = {
            "code": 0,
            "message": "success",
            "data": {
                "lat": 29.5223185,
                "lon": 106.3227816,
                "speed": 1.2  # 当前车辆速度
            }
        }

        # 向管道发送行为决策信息和 ego_data
        conn.send({"behavior_data_json_import": behavior_data_json_import, "ego_data": ego_data})
        #print("发送的数据:", behavior_data_json_import)

        # 模拟每隔一段时间发送一次
        time.sleep(1)


def speed_planning4_4(behavior, current_speed, road_speed_limit, is_in_lane_change, a_max, a_min, lane_change_speed):
    """
    根据行为决策信息进行速度规划，并考虑加速度限制。
    - behavior == 1: 提速至道路限速的80%
    - is_in_lane_change == True: 变道过程中逐步减速至40 km/h
    """
    if behavior == 1:  # 提速至道路限速的80%
        target_speed = 0.8 * road_speed_limit  # 道路限速的80%
        if current_speed < target_speed:
            # 计算加速度，限制最大加速度为a_max
            acceleration = min(target_speed - current_speed, a_max)  # 限制加速
            new_speed = current_speed + acceleration
        else:
            new_speed = target_speed  # 达到目标速度时保持
    elif is_in_lane_change:  # 如果在变道过程中
        target_speed = lane_change_speed  # 使用外部传入的变道目标速度

        if current_speed > target_speed:
            # 逐步减速至目标速度，限制最大减速度a_min
            deceleration = min(current_speed - target_speed, -a_min)  # 限制减速
            new_speed = current_speed - deceleration
        else:
            new_speed = target_speed  # 如果已经减速到目标速度，保持不变
    else:
        new_speed = current_speed  # 如果没有行为决策，保持当前速度

    return new_speed

def speed_planning(vertical_behavior, current_speed, road_speed_limit, is_in_lane_change, a_max, a_min, lane_change_speed, front_vehicle_speed=None):
    """
    根据行为决策信息进行速度规划，并考虑加速度限制。
    - vertical_behavior == 4: 加速
    - vertical_behavior == 5: 保持当前速度
    - vertical_behavior == 6: 减速
    - front_vehicle_speed: 如果前车速度有值，以前车速度作为期望速度
    - is_in_lane_change: 在变道过程中逐步减速至 lane_change_speed
    """
    # 1. 确定目标速度
    if front_vehicle_speed is None:
        # 有前车速度，目标速度受限于前车速度
        # target_speed = min(front_vehicle_speed, road_speed_limit)
        target_speed = 0.8 * road_speed_limit
    elif is_in_lane_change:
        # 变道过程中目标速度设置为变道速度
        target_speed = lane_change_speed
    elif vertical_behavior == 4:  # 加速
        # 道路限速的80%为目标速度
        target_speed = road_speed_limit
    elif vertical_behavior == 6:  # 减速
        # 减速目标速度为前车速度
        target_speed = front_vehicle_speed
    elif vertical_behavior == 5:  # 保持当前速度
        target_speed = current_speed
    else:  # front_vehicle_speed is not None
        # 进不到这里面来
        target_speed = current_speed

    # 2. 根据目标速度计算加速或减速
    if current_speed < target_speed:
        # 加速情况：限制最大加速度为 a_max
        acceleration = min(target_speed - current_speed, a_max)
        #print("加速度", acceleration)
        new_speed = current_speed + acceleration
    elif current_speed > target_speed:
        # 减速情况：限制最大减速度为 a_min
        deceleration = min(current_speed - target_speed, -a_min)  # a_min 是负值
        new_speed = current_speed - deceleration
    else:
        # 当前速度等于目标速度，保持不变
        new_speed = target_speed
    #print("前车速度", front_vehicle_speed, "自车速度", current_speed, "目标速度", target_speed)
    return new_speed, target_speed


def extract_local_path(global_path, current_index, change_lane_length=20):
    """
    从当前路径中提取指定长度的局部路径。

    :param global_path: 全局路径，表示所有路径点的列表。
    :param current_index: 当前路径索引，表示车辆当前位置。
    :param change_lane_length: 从当前位置开始提取的路径点数，默认为 20 个点。
    :return: 局部路径（包含指定长度的路径点）。
    """
    local_path = []
    # 确保当前索引不超出路径长度
    end_index = min(current_index + change_lane_length, len(global_path))
    local_path = global_path[current_index:end_index]

    # 如果局部路径为空，返回空路径
    if not local_path:
        print("局部路径为空！")
    return local_path

from math import sqrt

def cartesian_to_frenet(reference_path):
    """
    将给定的参考路径点集从笛卡尔坐标系转换到Frenet坐标系，仅考虑 (x, y) 转换为 (s, d)。

    reference_path: 参考路径点集，格式为 [(x1, y1, theta1, kappa1), (x2, y2, theta2, kappa2), ...]

    返回：
    frenetic_path：路径点集的Frenet坐标形式，格式为 [(s1, d1), (s2, d2), ...]
    """
    frenetic_path = []
    total_length = 0.0  # 用来记录弧长
    previous_point = reference_path[0]  # 第一个路径点
    prev_x, prev_y, _, _ = previous_point

    # 起始点的横向偏移 d 默认设置为 0
    frenetic_path.append((0.0, 0.0))

    # 遍历参考路径的每个点
    for i in range(1, len(reference_path)):
        # 获取当前路径点
        x, y, _, _ = reference_path[i]

        # 计算当前路径段的弧长增量
        segment_length = sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
        total_length += segment_length

        # 计算横向距离 d：使用叉积方法，计算点到路径的垂直距离
        dx = x - prev_x
        dy = y - prev_y
        norm = sqrt(dx**2 + dy**2)
        if norm == 0:
            d = 0.0  # 避免除以零
        else:
            perp_vector = np.array([-dy, dx])  # 路径切线的垂直向量
            d = np.dot(perp_vector, np.array([x - prev_x, y - prev_y])) / norm

        # 将路径点转换为 Frenet 坐标系，添加到列表中
        frenetic_path.append((total_length, d))

        # 更新前一个路径点
        prev_x, prev_y = x, y

    return frenetic_path

def lane_change_trajectory(points, s_ego, s_end, d_ego, d_end, num_points=100):
    """
    根据给定的 (s, d) 点集生成变道轨迹，返回变道轨迹的点集 (s, d)。

    参数
    ----------
    points : list of tuple
        车道中心线的 (s, d) 点集，形式为 [(s_1, d_1), (s_2, d_2), ...]。
    s_ego : float
        起始弧长。
    s_end : float
        终止弧长。
    d_ego : float
        起始横向位移。
    d_end : float
        终止横向位移。
    num_points : int, optional
        生成的轨迹点数量，默认为 100。

    返回
    -------
    list of tuple
        包含 (s, d(s)) 的轨迹点列表，表示弧长和横向位移。
    """
    # 提取 (s, d) 点
    s_values, d_values = zip(*points)

    # 设置矩阵 A 和向量 b
    A = np.array([
        [s_ego ** 5, s_ego ** 4, s_ego ** 3, s_ego ** 2, s_ego, 1],
        [5 * s_ego ** 4, 4 * s_ego ** 3, 3 * s_ego ** 2, 2 * s_ego, 1, 0],
        [20 * s_ego ** 3, 12 * s_ego ** 2, 6 * s_ego, 2, 0, 0],
        [s_end ** 5, s_end ** 4, s_end ** 3, s_end ** 2, s_end, 1],
        [5 * s_end ** 4, 4 * s_end ** 3, 3 * s_end ** 2, 2 * s_end, 1, 0],
        [20 * s_end ** 3, 12 * s_end ** 2, 6 * s_end, 2, 0, 0]
    ])
    b = np.array([d_ego, 0, 0, d_end, 0, 0])

    # 计算五次多项式的系数
    coefficients = np.linalg.solve(A, b)

    # 打印系数
    #print("拟合的五次多项式系数：", coefficients)

    # 生成变道轨迹
    s_values_new = np.linspace(s_ego, s_end, num_points)  # 弧长的值
    trajectory = [(s, _d(s, coefficients)) for s in s_values_new]

    return trajectory

def _d(s, coefficients):
    """
    根据五次多项式系数计算给定弧长 s 处的横向位移 d(s)。
    """
    return (coefficients[0] * s ** 5 +
            coefficients[1] * s ** 4 +
            coefficients[2] * s ** 3 +
            coefficients[3] * s ** 2 +
            coefficients[4] * s +
            coefficients[5])

from scipy.interpolate import interp1d

def frenet_to_cartesian(ref_path, frenet_path):
    """
    将 Frenet 路径转换为笛卡尔路径
    :param ref_path: 参考路径点集 [(x, y, theta, kappa), ...]
    :param frenet_path: Frenet 路径 [(s, d), ...]
    :return: 笛卡尔路径 [(x, y), ...]
    """
    # 提取参考路径的 x 和 y 坐标
    ref_x = np.array([p[0] for p in ref_path])
    ref_y = np.array([p[1] for p in ref_path])

    # 计算参考路径的弧长
    arc_lengths = np.cumsum(np.sqrt(np.diff(ref_x, prepend=ref_x[0])**2 + np.diff(ref_y, prepend=ref_y[0])**2))

    # 确保 arc_lengths 是严格递增的
    mask = np.diff(arc_lengths, prepend=arc_lengths[0]) > 1e-6
    ref_x = ref_x[mask]
    ref_y = ref_y[mask]
    arc_lengths = arc_lengths[mask]

    # 插值函数：根据弧长找到对应的参考坐标和方向
    x_interp = interp1d(arc_lengths, ref_x, kind='cubic', fill_value="extrapolate")
    y_interp = interp1d(arc_lengths, ref_y, kind='cubic', fill_value="extrapolate")

    # Frenet 转换输出路径
    cartesian_path = []

    # 遍历 Frenet 路径点 (s, d)
    for s, d in frenet_path:
        # 获取参考路径中的点
        x_ref = x_interp(s)
        y_ref = y_interp(s)

        # 计算参考路径切线方向 θ
        dx = x_interp(s + 1e-3) - x_interp(s - 1e-3)
        dy = y_interp(s + 1e-3) - y_interp(s - 1e-3)
        theta = np.arctan2(dy, dx)

        # 转换到笛卡尔坐标
        x = x_ref + d * np.cos(theta + np.pi / 2)
        y = y_ref + d * np.sin(theta + np.pi / 2)

        cartesian_path.append((x, y))

    return cartesian_path

def cartesian_to_path(xy_points):
    """
    将笛卡尔路径点集 (x, y) 转换为 (x, y, theta, kappa)。

    参数:
        xy_points (list of tuple): 输入路径点集，每个点是 (x, y)。

    返回:
        list of tuple: 输出路径点集，每个点是 (x, y, theta, kappa)。
    """
    n = len(xy_points)
    if n < 2:
        raise ValueError("路径点集至少需要两个点以计算 theta 和 kappa。")

    # 转换为 numpy 数组以便处理
    xy_points = np.array(xy_points)
    x = xy_points[:, 0]
    y = xy_points[:, 1]

    # 移除重复点
    unique_indices = np.unique(np.stack((x, y), axis=1), axis=0, return_index=True)[1]
    unique_indices = sorted(unique_indices)  # 保持顺序
    x = x[unique_indices]
    y = y[unique_indices]

    # 一阶导数 (有限差分)
    dx = np.gradient(x)
    dy = np.gradient(y)

    # 二阶导数 (有限差分)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 计算 theta (方向角)
    theta = np.arctan2(dy, dx)

    # 计算 kappa (曲率)
    kappa = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)

    # 组装结果
    path_with_theta_kappa = [(x[i], y[i], theta[i], kappa[i]) for i in range(len(x))]
    return path_with_theta_kappa


# TODO 代码修改日志
'''
12.9  记录
基于  实车—Carla11.24模块调试4.py   修改，尝试加入速度规划，在控制行为决策比例（0，1，2）比例（1：3：1）
random_behavior = random.choices([0, 1, 2], weights=[1, 3, 1])[0]  # 根据权重选择 0, 1 或 2
让大部分时间都保持直行的情况下，进行速度规划。暂不考虑感知，只在行为 1 直行，进行速度规划，行驶至道路限速的 80% 速度行驶
在有变道规划时，输出变道轨迹之后也进行速度规划，变道过程中持续减速至 40km/h 进行变道，变道之后直行，又重新提速，速度规划

纵向加速度
a_max = 3 m/s^2
a_min = -4.5 m/s^2

存在问题：
1、直线路径太短了，只有232个轨迹点（已改进现在有1154个点）

12.18  记录
1、 4.3 改进代码，针对纵向决策，延伸出 vertical_behavior 和 front_vehicle_speed 两字段， vertical_behavior 表示纵向决策 4 加速  5 保持当前速度  6 减速，front_vehicle_speed 表示前车速度
2、 对 4.3 进行修改 4.4 速度规划，以前车速度为基准

'''

if __name__ == "__main__":
    # 连接到CARLA服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 获取世界和地图
    world = client.get_world()
    map = world.get_map()

    # 设置起点和终点
    # start_location0 = carla.Location(x=-1019.90, y=-907.74, z=2)
    # start_location1 = carla.Location(x=-1023.32, y=-908.26, z=2)
    # end_location0 = carla.Location(x=-1237.45, y=415.27, z=2)
    # end_location1 = carla.Location(x=-1241.03, y=414.31, z=2)

    # 新 L 型地图
    start_location0 = carla.Location(x=-1019.90, y=-907.74, z=2)
    start_location1 = carla.Location(x=-1023.32, y=-908.26, z=2)
    # end_location0 = carla.Location(x=-1079.78, y=-440.00, z=2)
    # end_location1 = carla.Location(x=-1083.39, y=-440.26, z=2)

    end_location0 = carla.Location(x=-1235.31, y=1404.68, z=2)
    end_location1 = carla.Location(x=-1238.68, y=1404.57, z=2)

    # 获取起点和终点的最近路点
    start_waypoint0 = map.get_waypoint(start_location0, project_to_road=True)
    end_waypoint0 = map.get_waypoint(end_location0, project_to_road=True)
    start_waypoint1 = map.get_waypoint(start_location1, project_to_road=True)
    end_waypoint1 = map.get_waypoint(end_location1, project_to_road=True)

    # 获取路径规划器并计算路径
    global_route_plan = global_path_planner(world_map=map, sampling_resolution=2)
    pathway0 = global_route_plan.search_path_way(origin=start_waypoint0.transform.location,
                                                 destination=end_waypoint0.transform.location)
    pathway1 = global_route_plan.search_path_way(origin=start_waypoint1.transform.location,
                                                 destination=end_waypoint1.transform.location)

    # 将路径点转换为 (x, y, theta, kappa) 格式
    global_frenet_path0 = waypoint_list_2_target_path(pathway0)
    global_frenet_path1 = waypoint_list_2_target_path(pathway1)

    #print("路径0长度", len(global_frenet_path0))
    #print("路径1长度", len(global_frenet_path1))

    # 创建并初始化车辆
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_point = start_waypoint0.transform
    spawn_point.location.z = 1.0
    model3_actor = world.try_spawn_actor(vehicle_bp, spawn_point)
    if model3_actor is None:
        raise ValueError("Vehicle could not be spawned. Check spawn point and blueprint.")

    # 控制器参数
    max_speed = 22   # m/s
    vehicle_para = (1.015, 2.910 - 1.015, 1412, -148970, -82204, 1537)
    controller = "MPC_controller"
    Controller = Vehicle_control(ego_vehicle=model3_actor, vehicle_para=vehicle_para,
                                 pathway=global_frenet_path0, controller_type=controller)

    # 创建管道
    parent_conn, child_conn = Pipe()

    # 启动发送器进程
    sender_process = Process(target=behavior_data_sender, args=(parent_conn,))
    sender_process.start()

    # 初始化路径状态
    current_path = global_frenet_path0
    global_index = 0
    is_in_lane_change = False
    transition_path = None
    target_path = None

    # 初始化行为变量，确保不会报错
    behavior = 1  # 假设行为初始化为 "0" 或者 "保持当前车道" 的行为
    global_index_lane = 0
    # 设置提取的路径长度 10 个点为 10 * 2 = 20 m
    change_lane_length = 10

    # 设置变道速度
    lane_change_speed = 11  # 变道时目标速度40 km/h

    try:
        while True:
            # 从管道接收数据
            if child_conn.poll():
                data = child_conn.recv()
                #print("接收到的数据:", data)

                behavior_data_json_import = data.get("behavior_data_json_import", {})
                ego_data = data.get("ego_data", {})
                #print("行为数据:", behavior_data_json_import)
                #print("Ego数据:", ego_data)

                # 提取行为优先级信息
                behavior_priority = behavior_data_json_import.get("result", {}).get("vehicle_behavior_priority", {})
                #print("提取的行为优先级:", behavior_priority)

                # 获取当前车辆的行为
                behavior = behavior_priority.get("QD1E003P", {}).get("behavior", None)
                #print("当前行为:", behavior)

                # 确认行为决策目标路径
                if behavior == 0:
                    new_target_path = global_frenet_path0  # 切换到左车道
                elif behavior == 2:
                    new_target_path = global_frenet_path1  # 切换到右车道
                else:
                    new_target_path = current_path  # 保持当前车道

                if not is_in_lane_change and new_target_path != current_path:
                    global_index_lane = global_index
                    #print("变道前全局索引为：", global_index_lane)
                    # 检查是否接近路径终点
                    if global_index >= len(current_path) - 3:
                        #print("接近路径终点，跳过变道规划")
                        continue

                    #print(f"当前索引: {global_index}, 当前路径长度: {len(current_path)}, 目标路径长度: {len(new_target_path)}")

                    # 1、提取局部路径，从当前路径的索引开始，取 20 个点，40m，格式为 x, y, theta, k
                    # change_lane_length = 10  # 设置提取的路径长度
                    local_path = extract_local_path(current_path, global_index, change_lane_length)

                    # 2、执行变道逻辑，运用waypoint_list_2_target_path函数转换成 x, y, theta, k 形式
                    # print(local_path)
                    # local_frenet_path = waypoint_list_2_target_path(local_path)

                    # 3、将笛卡尔路径转换为Frenet路径，artesian_to_frenet 函数，[(x1, y1, theta1, kappa1), (x2, y2, theta2, kappa2), ...] 转换成 [(s1, d1), (s2, d2) 格式
                    frenetic_path = cartesian_to_frenet(local_path)
                    #print(frenetic_path)

                    # 设置边界条件
                    right = 3.5
                    left = -3.5
                    # choices = [right, left]

                    s_ego = 0  # 起始弧长
                    s_end = 20  # 终止弧长
                    d_ego = 0  # 起始横向位移
                    if behavior == 0:
                        d_end = left
                    else:
                        # behavior == 2  右变道
                        d_end = right
                    # d_end = random_choice = random.choice(choices)  # 终止横向位移，3米表示向左变道，-3米表示向右变道

                    # 4、生成变道轨迹，lane_change_trajectory 函数，生成变道轨迹
                    trajectory = lane_change_trajectory(frenetic_path, s_ego, s_end, d_ego, d_end)
                    #print("变道轨迹", trajectory)

                    # 执行路径转换
                    cartesian_path = frenet_to_cartesian(local_path, trajectory)
                    #print("s，d转换成x，y", cartesian_path)

                    transition_path = cartesian_to_path(cartesian_path)
                    #print("x，y转换成x, y, theta, k", transition_path)



                    # 执行变道规划
                    target_path = new_target_path
                    # transition_path_xy, target_path = trajectory_planning(global_index, current_path, new_target_path)
                    # transition_path = xy_list_2_target_path(transition_path_xy)

                    if transition_path:
                        #print("生成变道轨迹", transition_path)
                        current_path = transition_path
                        global_index = 0  # 重置索引到变道轨迹的起点
                        is_in_lane_change = True

                        # 实例化控制器，使其跟踪变道轨迹
                        Controller = Vehicle_control(
                            ego_vehicle=model3_actor,
                            vehicle_para=vehicle_para,
                            pathway=current_path,  # 设置为变道轨迹
                            controller_type=controller
                        )
                        draw_debug_path(world, transition_path, color=(255, 0, 0))  # 绘制变道轨迹

            # 获取当前速度
            current_speed = model3_actor.get_velocity().length()
            #print("当前速度为：", current_speed * 3.6)

            # 调用速度规划函数
            new_speed = speed_planning4_4(behavior, current_speed, max_speed, is_in_lane_change, a_max=3, a_min=-4.5, lane_change_speed=lane_change_speed)
            #print("速度规划的速度：", new_speed * 3.6)

            # 控制车辆沿当前路径行驶
            control = Controller.run_step(target_speed=new_speed * 3.6)
            model3_actor.apply_control(control)

            # 更新索引并检测终点
            vehicle_loc = model3_actor.get_transform().location
            global_index = find_nearest_path_point(vehicle_loc.x, vehicle_loc.y, current_path)
            #print("当前索引是", global_index)

            # 检查是否完成变道
            if is_in_lane_change and global_index >= len(current_path) - 1:
                current_path = target_path
                #print("变道完成，切换到目标路径")
                global_index = global_index_lane + change_lane_length
                # 将当前索引往后的路径点作为新的轨迹、拼接轨迹 current_path_combination
                current_path_combination = target_path[global_index + 1:]
                #print("变道后新轨迹的起点为", current_path_combination[0])
                # global_index = 0  # 重置索引到目标路径的起点
                is_in_lane_change = False

                # 更新控制器轨迹
                Controller = Vehicle_control(
                    ego_vehicle=model3_actor,
                    vehicle_para=vehicle_para,
                    pathway=current_path_combination,  # 设置为目标路径
                    controller_type=controller
                )
                draw_debug_path(world, current_path_combination, color=(0, 255, 0))  # 绘制目标路径

    except KeyboardInterrupt:
        print("Manual interrupt received. Stopping simulation.")
    finally:
        if model3_actor is not None:
            model3_actor.destroy()
            #print("Vehicle destroyed.")
