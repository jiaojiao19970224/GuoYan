#!/usr/bin/env python
# coding: utf-8

# 导入所需的库
#import asyncio  # 用于异步操作
import json  # 用于JSON数据处理
import joblib  # 用于加载机器学习模型
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
from scipy.signal import butter, filtfilt  # 用于信号滤波
import math  # 用于数学计算
#import random  # 用于生成随机数
#import time  # 用于时间相关操作
#import requests  # 用于HTTP请求
#from requests.exceptions import RequestException  # 用于处理请求异常


def lowpass_filter(data, cutoff=0.1, fs=10.0, order=2):
    """
    实现低通滤波器
    参数:
        data: 输入数据
        cutoff: 截止频率
        fs: 采样频率
        order: 滤波器阶数
    返回:
        滤波后的数据
    """
    if len(data) <= order * 3:
        print("数据长度太短，无法进行滤波，返回原始数据。")
        return data
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    result = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(result[0], result[1], data)  # 直接使用 result 的索引
    return y


def prepare_data_for_prediction(df):
    """
    准备用于预测的数据
    参数:
        df: 包含原始数据的DataFrame
    返回:
        处理后的DataFrame
    """
    # 将相关列转换为数值类型
    df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
    df['heading'] = pd.to_numeric(df['heading'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['lat'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['lon'], errors='coerce')
    df['type'] = pd.to_numeric(df['type'], errors='coerce')

    # 对速度和航向进行低通滤波
    df['speed_filtered'] = lowpass_filter(df['speed'], cutoff=0.1, fs=10.0, order=2)
    df['heading_filtered'] = lowpass_filter(df['heading'], cutoff=0.1, fs=10.0, order=2)

    # 计算加速度和航向变化率
    df['acceleration'] = df['speed_filtered'].diff() * 10
    df['heading_rate_of_change'] = df['heading_filtered'].diff() * 10

    # 填充缺失值
    df['acceleration'].fillna(0, inplace=True)
    df['heading_rate_of_change'].fillna(0, inplace=True)

    # 计算横向加速度
    df['lateral_acceleration'] = df['speed_filtered'] * (df['heading_rate_of_change'] * np.pi / 180)
    return df


def predict_intent(df, xgb_model, scaler):
    """
    使用训练好的模型预测车辆意图
    参数:
        df: 处理后的数据
        xgb_model: XGBoost模型
        scaler: 标准化器
    返回:
        预测的意图
    """
    features = ['speed_filtered', 'heading_filtered', 'latitude', 'longitude',
                'acceleration', 'heading_rate_of_change', 'lateral_acceleration']
    X = df[features]
    X_scaled = scaler.transform(X)
    y_pred_intent = xgb_model.predict(X_scaled)
    return y_pred_intent


def determine_target_lane(lane_index, lane_num):
    """
    确定可能的目标车道
    参数:
        lane_index: 当前车道索引
        lane_num: 车道总数
    返回:
        可能的目标车道列表
    """
    potential_target_lanes = []
    if lane_index > 1:
        potential_target_lanes.append(lane_index - 1)  # 左侧车道
    if lane_index < lane_num:
        potential_target_lanes.append(lane_index + 1)  # 右侧车道
    return potential_target_lanes


def calculate_lane_change_necessity(S_PV, v_SV, t_THW):
    """
    计算换道必要性
    参数:
        S_PV: 与前车的距离
        v_SV: 本车速度
        t_THW: 时距阈值
    返回:
        换道必要性指标
    """
    if v_SV == 0:
        return float('inf')
    return S_PV - v_SV * t_THW


def decide_lane_change_intent(trend_speed_benefit, trend_space_benefit, f_necessity,
                              current_lane, potential_target_lanes,
                              lane_num, current_speed, v_desired,
                              speed_threshold=3.0, space_threshold=8.0, necessity_threshold=5.0):
    """
    决策换道意图
    参数:
        trend_speed_benefit: 速度收益趋势
        trend_space_benefit: 空间收益趋势
        f_necessity: 换道必要性
        current_lane: 当前车道
        potential_target_lanes: 可能的目标车道
        lane_num: 车道总数
        current_speed: 当前速度
        v_desired: 期望速度
        speed_threshold: 速度阈值
        space_threshold: 空间阈值
        necessity_threshold: 必要性阈值
    返回:
        决策的换道意图(0:左换道, 1:保持车道, 2:右换道)
    """
    best_intent = 1  # 默认保持车道
    best_benefit = -float('inf')

    # 评估保持车道的合理性
    keep_lane_benefit = abs(current_speed - v_desired)
    if keep_lane_benefit < speed_threshold:
        return 1

    # 根据车道位置确定允许的换道意图
    allowed_intents = []
    if current_lane == 1:
        allowed_intents = [1, 2]  # 最左侧车道只能保持或右换道
    elif current_lane == lane_num:
        allowed_intents = [0, 1]  # 最右侧车道只能保持或左换道
    else:
        allowed_intents = [0, 1, 2]  # 中间车道可以左右换道或保持

    # 评估换道收益
    for target_lane in potential_target_lanes:
        if (trend_speed_benefit > speed_threshold and
                trend_space_benefit > space_threshold and
                f_necessity > necessity_threshold):

            benefit = trend_speed_benefit + trend_space_benefit + f_necessity
            if benefit > best_benefit:
                if target_lane < current_lane and 0 in allowed_intents:
                    best_intent = 0  # 左换道
                    best_benefit = benefit
                elif target_lane > current_lane and 2 in allowed_intents:
                    best_intent = 2  # 右换道
                    best_benefit = benefit

    # 如果换道收益不明显，保持车道
    if best_benefit < necessity_threshold:
        best_intent = 1

    return best_intent


def trend_analysis(values, window_size):
    """
    分析数据趋势
    参数:
        values: 数据值列表
        window_size: 窗口大小
    返回:
        趋势值
    """
    if len(values) >= window_size:
        return np.mean(values[-window_size:])
    else:
        return np.mean(values)


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    计算两个经纬度点之间的距离（单位：米）
    使用Haversine公式计算球面两点距离
    """
    R = 6371000  # 地球平均半径（米）

    # 将经纬度转换为弧度
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine公式
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def find_leading_vehicle(current_vehicle, all_vehicles, same_lane_threshold=2):
    """
    寻找前车信息
    参数:
        current_vehicle: 当前车辆数据
        all_vehicles: 所有车辆数据
        same_lane_threshold: 认为是同一车道的偏差阈值
    返回:
        leading_vehicle: 前车数据
        distance: 与前车的距离
    """
    leading_vehicle = None
    min_distance = float('inf')
    current_lat = current_vehicle['lat']
    current_lon = current_vehicle['lon']
    current_lane = current_vehicle['laneIndex']
    current_heading = current_vehicle['heading']

    for vehicle in all_vehicles:
        # 跳过自身
        if vehicle['vehicleId'] == current_vehicle['vehicleId']:
            continue

        # 检查是否在同一车道
        if abs(vehicle['laneIndex'] - current_lane) > same_lane_threshold:
            continue

        # 计算距离
        distance = calculate_distance(current_lat, current_lon,
                                      vehicle['lat'], vehicle['lon'])

        # 检查是否在前方（根据航向角判断）
        relative_heading = ((vehicle['heading'] - current_heading + 180) % 360) - 180
        if abs(relative_heading) > 45:  # 考虑45度范围内的车辆
            continue

        if distance < min_distance:
            leading_vehicle = vehicle
            min_distance = distance

    return leading_vehicle, min_distance


class IntentRecognitionModule:
    """
    意图识别模块类
    """

    def __init__(self, xgb_model, scaler):
        """
        初始化意图识别模块
        参数:
            xgb_model: XGBoost模型
            scaler: 标准化器
        """
        self.xgb_model = xgb_model
        self.scaler = scaler
        self.vehicle_history_data = {}
        self.last_lane_change_timestamp = {}  # 改用timestamp记录上次换道时间
        self.last_lane_index = {}  # 记录上次的车道号

    def process_vehicle_data(self, vehicle_id, realtime_data):
        """
        处理车辆实时数据
        参数:
            vehicle_id: 车辆ID
            realtime_data: 实时数据
        返回:
            处理结果字典
        """
        # 检查数据有效性
        if ('data' not in realtime_data or
                'targets' not in realtime_data['data'] or
                len(realtime_data['data']['targets']) == 0):
            return {
                "targets": [],
                "timestamp": realtime_data.get("timestamp"),
                "fetchedData": realtime_data
            }

        timestamp = realtime_data['data'].get('timestamp')
        df = pd.DataFrame(realtime_data['data']['targets'])
        df['type'] = pd.to_numeric(df['type'], errors='coerce')
        results = []

        if not df.empty and vehicle_id in df['vehicleId'].values:
            host_vehicle_heading = df.loc[df['vehicleId'] == vehicle_id, 'heading'].values[0]
            other_vehicles = df.to_dict('records')

            for _, row in df.iterrows():
                # 只处理特定类型的车辆
                if row['type'] not in [2, 5]:
                    continue

                # 处理车辆ID
                if pd.isna(row['vehicleId']):
                    row['vehicleId'] = row['uuid']

                if 'uuid' in row.index:
                    row.drop('uuid', inplace=True)

                # 处理目标车辆
                if row['vehicleId'] == vehicle_id:
                    # 记录当前数据
                    current_data = {
                        "lat": float(row['lat']),
                        "lon": float(row['lon']),
                        "speed": float(row['speed']),
                        "timestamp": timestamp,
                        "laneIndex": int(row['laneIndex']),
                        "laneNum": int(row['laneNum'])
                    }

                    # 更新历史数据
                    if vehicle_id not in self.vehicle_history_data:
                        self.vehicle_history_data[vehicle_id] = []
                    self.vehicle_history_data[vehicle_id].append(current_data)

                    # 限制历史数据长度
                    if len(self.vehicle_history_data[vehicle_id]) > 50:
                        self.vehicle_history_data[vehicle_id] = self.vehicle_history_data[vehicle_id][-50:]

                    # 提取当前状态
                    current_lat = current_data['lat']
                    current_lon = current_data['lon']
                    current_speed = current_data['speed']
                    current_lane = current_data['laneIndex']
                    lane_num = current_data['laneNum']

                    # 确定潜在目标车道
                    potential_target_lanes = determine_target_lane(current_lane, lane_num)
                    v_desired = 50 / 3.6  # 期望速度(m/s)
                    t_THW = 1.5  # 时距阈值

                    # 查找前车信息
                    leading_vehicle, distance_to_leading = find_leading_vehicle(row, other_vehicles)

                    # 计算相关参数
                    if leading_vehicle is not None:
                        v_PV = leading_vehicle['speed']  # 前车速度
                        S_PV = distance_to_leading  # 与前车距离
                    else:
                        # 无前车时的默认值
                        v_PV = v_desired
                        S_PV = 100  # 设置一个较大的安全距离

                    # 计算换道必要性
                    f_necessity = calculate_lane_change_necessity(S_PV, v_PV - current_speed, t_THW)

                    # 分析趋势
                    speeds = [data_point['speed'] for data_point in self.vehicle_history_data[vehicle_id]]
                    # 使用实际数据计算趋势
                    if len(self.vehicle_history_data[vehicle_id]) >= 2:
                        space_benefits = []
                        speed_benefits = []
                        for i in range(len(self.vehicle_history_data[vehicle_id]) - 1):
                            space_benefits.append(S_PV)
                            speed_benefits.append(v_desired - speeds[i])
                        trend_space_benefit = trend_analysis(space_benefits, 5)
                        trend_speed_benefit = trend_analysis(speed_benefits, 5)
                    else:
                        trend_space_benefit = S_PV
                        trend_speed_benefit = v_desired - current_speed

                    # 初始化车道记录
                    if vehicle_id not in self.last_lane_index:
                        self.last_lane_index[vehicle_id] = current_lane
                        self.last_lane_change_timestamp[vehicle_id] = 0

                    # 检查是否发生了实际换道（车道号发生变化）
                    if current_lane != self.last_lane_index[vehicle_id]:
                        self.last_lane_change_timestamp[vehicle_id] = timestamp  # 更新换道时间戳
                        self.last_lane_index[vehicle_id] = current_lane  # 更新车道记录

                    # 计算自上次换道后经过的时间
                    cooling_time = (timestamp - self.last_lane_change_timestamp[vehicle_id]) / 1000.0  # 转换为秒

                    # 如果在换道冷却期内
                    if cooling_time < 10.0:  # 冷却时间
                        predicted_intent = 1  # 强制保持车道
                    else:
                        print(f"前车距离: {S_PV:.2f}m, 前车速度: {v_PV:.2f}m/s, "
                              f"换道必要性: {f_necessity:.2f}")

                     # 决策换道意图
                        predicted_intent = decide_lane_change_intent(
                        trend_speed_benefit, trend_space_benefit, f_necessity,
                        current_lane, potential_target_lanes,
                        lane_num, current_speed, v_desired
                    )

                    results.append({
                        "vehicleId": row['vehicleId'],
                        "vehicleIntention": predicted_intent,
                        "isHostVehicle": True
                    })

                else:
                    # 处理周车
                    vehicle_data_df = pd.DataFrame([row])
                    vehicle_data_df = prepare_data_for_prediction(vehicle_data_df)
                    y_pred_intent = predict_intent(vehicle_data_df, self.xgb_model, self.scaler)
                    row['vehicleIntention'] = y_pred_intent[0]
                    results.append({
                        "vehicleId": row['vehicleId'],
                        "vehicleIntention": int(row['vehicleIntention']),
                        "isHostVehicle": False
                    })

        # 返回处理结果
        response = {
            "targets": results,
            "timestamp": timestamp,
            "fetchedData": realtime_data

        }
        return response

def main():

        xgb_model = joblib.load('model/intent_recognition_model_2.pkl')
        scaler = joblib.load('model/scaler2.pkl')

        intent_module = IntentRecognitionModule(xgb_model, scaler)

        vehicle_id = "12345"
        realtime_data = {
            "data": {
                "timestamp": 1633046400000,
                "targets": [
                    {
                        "vehicleId": "12345",
                        "lat": 31.2304,
                        "lon": 121.4737,
                        "speed": 15.0,
                        "heading": 90.0,
                        "type": 2
                    },
                    {
                        "vehicleId": "67890",
                        "lat": 31.2310,
                        "lon": 121.4740,
                        "speed": 10.0,
                        "heading": 90.0,
                        "type": 2
                    }
                ]
            }
        }

        realtime_data1 = """
            {
                "code": 0,
                "data": {
                    "objNum": 4,
                    "targets": [
                        {
                            "crossId": "73",
                            "heading": 277.6761,
                            "height": 160,
                            "lat": 29.6098089,
                            "length": 459,
                            "lon": 106.2906199,
                            "roadSegId": null,
                            "speed": 0.099999994,
                            "srcType": 3,
                            "type": 2,
                            "uuid": "97184f98d3e2089f89e4eb94",
                            "vehicleId": null,
                            "width": 179,
                            "laneIndex": 1,
                            "laneNum": 5
                        },
                        {
                            "crossId": "73",
                            "heading": 7.9404,
                            "height": 179,
                            "lat": 29.6096309,
                            "length": 160,
                            "lon": 106.2906075,
                            "roadSegId": "S73down",
                            "speed": 0.76,
                            "srcType": 3,
                            "type": 1,
                            "uuid": "4956b61c3bf38f4889e4ff1c",
                            "vehicleId": null,
                            "width": 80,
                            "laneIndex": 1,
                            "laneNum": 1
                        },
                        {
                            "crossId": "73",
                            "heading": 184.9875,
                            "height": null,
                            "lat": 29.6094882,
                            "length": null,
                            "lon": 106.2905825,
                            "roadSegId": "S73down",
                            "speed": 0,
                            "srcType": 1,
                            "type": 2,
                            "uuid": "1a04ab92e5af62ae89e117e4",
                            "vehicleId": "R1000004",
                            "width": null,
                            "laneIndex": 4,
                            "laneNum": 4
                        },
                        {
                            "crossId": "73",
                            "heading": 17.3516,
                            "height": 160,
                            "lat": 29.6096294,
                            "length": 459,
                            "lon": 106.2906078,
                            "roadSegId": "S73down",
                            "speed": 1.2099999,
                            "srcType": 3,
                            "type": 2,
                            "uuid": "b557929ad3e5746089e4b098",
                            "vehicleId": null,
                            "width": 179,
                            "laneIndex": 1,
                            "laneNum": 1
                        }
                    ],
                    "timestamp": 1728890341994
                },
                "message": "success"
            }
            """
        vehicle_id1 = 'R1000004'
        realtime_data1 = json.loads(realtime_data1)
        output = intent_module.process_vehicle_data(vehicle_id1, realtime_data1)

        print(json.dumps(output, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
