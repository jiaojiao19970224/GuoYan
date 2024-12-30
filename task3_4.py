import requests
import math
from math import radians, sin, cos, sqrt, atan2
import json
import numpy as np


class Task3_4:
    def __init__(self, host_vehicle_id, G=9.8, R=1, M=5000, max_brake_distance=50):
        self._G = G  # 重力加速度常量
        self._R = R  # 常量参数
        self._M = M  # 等效质量
        self._max_brake_distance = max_brake_distance  # 最大刹车距离
        self._host_vehicle_id = host_vehicle_id  # 主车ID
        self._vehicle_info = {}  # 车辆信息字典
        self._topological_set = {}  # 拓扑关系集合
        self._coop_set = {}  # 协作方集合
        self._type_list = {"2"}  # 对象类型集合
        self._host_vehicle_intention = 0  # 主车意图
        self._vehicles_with_intentions = []  # 包含意图的车辆集合

    def set_type_list(self, type_list):
        self._type_list = type_list

    def get_type_list(self):
        return self._type_list

    # 设置和获取 host_vehicle_id
    def get_host_vehicle_id(self):
        return self._host_vehicle_id

    def set_host_vehicle_id(self, value):
        self._host_vehicle_id = value

    # 设置和获取 vehicle_info
    def get_vehicle_info(self):
        return self._vehicle_info

    def set_vehicle_info(self, value):
        self._vehicle_info = value

    # 设置和获取 host_vehicle_intention
    def get_host_vehicle_intention(self):
        return self._host_vehicle_intention

    def set_host_vehicle_intention(self, value):
        self._host_vehicle_intention = value

    # 设置和获取 vehicles_with_intentions
    def get_vehicles_with_intentions(self):
        return self._vehicles_with_intentions

    def set_vehicles_with_intentions(self, value):
        self._vehicles_with_intentions = value

    # 设置和获取 coop_set
    def get_coop_set(self):
        return self._coop_set

    def set_coop_set(self, value):
        self._coop_set = value

    # 设置和获取 topological_set
    def get_topological_set(self):
        return self._topological_set

    def set_topological_set(self, value):
        self._topological_set = value

    def determine_target_lane(self, host_lane_index, host_intention):
        if host_intention == 1:  # 直行
            return host_lane_index
        elif host_intention == 0:  # 左转
            return host_lane_index - 1
        elif host_intention == 2:  # 右转
            return host_lane_index + 1

    def haversine(self, lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = 6371000 * c  # 计算地球表面距离，单位为米
        return distance


    def get_coopeset_1225(self):
        coop_set = set()
        host_intention = self._host_vehicle_intention
        targetLane = self.get_target_lane()
        if host_intention == 1:
            coop_set.add(self._host_vehicle_id)
        else:
            front_id, rear_id = self.find_closest_vehicle_1(self._vehicle_info, self._host_vehicle_id, targetLane)
            coop_set.add(self._host_vehicle_id)
            if front_id is not None:
                coop_set.add(front_id)
            if rear_id is not None:
                coop_set.add(rear_id)
        self._coop_set = coop_set
        return coop_set

    def get_surrounding_vehicles(self):
        """
        获取主车周围车辆列表
        """
        host_vehicle_info = self._vehicle_info.get(self._host_vehicle_id)
        if not host_vehicle_info:
            # print("未找到主车信息")
            return []

        host_lon = host_vehicle_info["lon"]
        host_lat = host_vehicle_info["lat"]
        host_vehicle_heading = host_vehicle_info["heading"]

        result = []
        for vehicle in self._vehicles_with_intentions:
            vehicle_id = vehicle["vehicle_id"]
            agent_info = self._vehicle_info.get(vehicle_id)
            if not agent_info:
                # print(f"未找到车辆 {vehicle_id} 的信息")
                continue

            agent_lon = agent_info["lon"]
            agent_lat = agent_info["lat"]
            agent_vehicle_heading = agent_info['heading']
            angle_diff = abs(host_vehicle_heading - agent_vehicle_heading)
            distance_of_host_and_agent = self.haversine(host_lon, host_lat, agent_lon, agent_lat)

            if distance_of_host_and_agent < self._max_brake_distance and vehicle_id != self._host_vehicle_id and (
                    0 <= angle_diff <= 60):
                result.append(vehicle_id)

        # print(f"周围车辆: {result}")
        return result

    def calculate_real_threshold(self, agent_vehicle_id):
        """
        计算主车在周围车辆处的真实场强值
        """
        host_vehicle_info = self._vehicle_info.get(self._host_vehicle_id)
        if not host_vehicle_info:
            return None
        host_vehicle_speed = host_vehicle_info['speed']
        host_vehicle_lon = host_vehicle_info['lon']
        host_vehicle_lat = host_vehicle_info['lat']
        agent_info = self._vehicle_info.get(agent_vehicle_id)
        if not agent_info:
            return None
        agent_vehicle_lon = agent_info['lon']
        agent_vehicle_lat = agent_info['lat']
        real_distance = self.haversine(host_vehicle_lon, host_vehicle_lat, agent_vehicle_lon, agent_vehicle_lat)
        if real_distance == 0:
            real_distance = 1e-6
        real_threshold = (self._G * self._M * self._R / real_distance) * math.exp(host_vehicle_speed)
        return real_threshold

    def calculate_min_threshold(self):
        """
        计算安全阈值边界
        """
        host_vehicle_speed = self._vehicle_info.get(self._host_vehicle_id)['speed']
        min_safe_threshold = (self._G * self._M * self._R / self._max_brake_distance) * math.exp(host_vehicle_speed)
        return min_safe_threshold

    def get_coope_set(self):
        """
        获取协作方集合
        """
        coope_set = set()
        host_vehicle_id = self._host_vehicle_id
        coope_set.add(host_vehicle_id)

        surrounding_vehicles = self.get_surrounding_vehicles()
        for vehicle_id in surrounding_vehicles:
            real_thre = self.calculate_real_threshold(vehicle_id)
            min_thre = self.calculate_min_threshold()

            # 如果阈值条件满足，则将车辆添加到协作方集合
            if real_thre > min_thre:
                coope_set.add(vehicle_id)

        self._coop_set = coope_set
        return coope_set

    def get_keyVehicles_set(self):
        key_vehicle_set = set()
        if self.get_host_vehicle_intention() != 1:
            front_id, rear_id = self.find_closest_vehicle_1(self._vehicle_info, self._host_vehicle_id, self._vehicle_info.get(self._host_vehicle_id)["laneIndex"])
            key_vehicle_set.add(front_id)
        return key_vehicle_set

    def get_target_lane(self):
        """
        意图匹配目标车道
        :return:
        """
        host_vehicle_info = self._vehicle_info.get(self._host_vehicle_id)
        host_vehicle_lane = host_vehicle_info["laneIndex"]
        host_intention = self.get_host_vehicle_intention()
        if host_intention == 1:  # 直行
            return host_vehicle_lane
        elif host_intention == 0:  # 左转
            return host_vehicle_lane - 1
        elif host_intention == 2:  # 右转
            return host_vehicle_lane + 1

    # 计算两点间的距离和方位角
    def haversine1(self, lat1, lon1, lat2, lon2):
        R = 6371  # 地球半径（单位：公里）
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        distance = R * c

        # 计算方位角
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        initial_bearing = math.atan2(y, x)

        initial_bearing = (math.degrees(initial_bearing) + 360) % 360  # 0-360度
        return distance, initial_bearing

    def find_closest_vehicle_1(self, vehicle_info, vehicle_id_a, targetLane):
        vehicle_a = vehicle_info[vehicle_id_a]
        lat_a, lon_a, heading_a = vehicle_a['lat'], vehicle_a['lon'], vehicle_a['heading']

        closest_vehicle_id = None
        closest_distance = float('inf')
        closest_vehicle_id_rear = None
        closest_distance_rear = float('inf')

        for vehicle_id_b, vehicle_b in vehicle_info.items():
            if abs(vehicle_a['heading'] - vehicle_b['heading']) < 20 and vehicle_b['laneIndex'] == targetLane:
                if vehicle_id_b == vehicle_id_a:
                    continue

                lat_b, lon_b, heading_b = vehicle_b['lat'], vehicle_b['lon'], vehicle_b['heading']

                # 计算车A与车B的距离和方位角
                distance, bearing = self.haversine1(lat_a, lon_a, lat_b, lon_b)


                heading_a_min = (heading_a - 90) % 360
                heading_a_max = (heading_a + 90) % 360

                if heading_a_min < heading_a_max:
                    # 如果航向角范围不跨越0度，直接判断
                    if heading_a_min <= bearing <= heading_a_max:
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_vehicle_id = vehicle_id_b
                    else:
                        if distance < closest_distance_rear:
                            closest_distance_rear = distance
                            closest_vehicle_id_rear = vehicle_id_b
                else:
                    # 如果航向角范围跨越0度，分两段判断
                    if bearing >= heading_a_min or bearing <= heading_a_max:
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_vehicle_id = vehicle_id_b
                    else:
                        if distance < closest_distance_rear:
                            closest_distance_rear = distance
                            closest_vehicle_id_rear = vehicle_id_b

        return closest_vehicle_id, closest_vehicle_id_rear


if __name__ == '__main__':
    # 输入逻辑
    host_vehicle_id = 'QD1E002A'

    # 创建类实例
    task = Task3_4(host_vehicle_id)

    # 输入车辆信息
    vehicle_info_input = input("请输入车辆信息(格式为JSON字符串): ")
    task.set_vehicle_info(json.loads(vehicle_info_input))

    # 输入主车意图
    host_vehicle_intention_input = 0
    task.set_host_vehicle_intention(host_vehicle_intention_input)

    # 输入车辆意图集合
    vehicles_with_intentions_input = input("请输入车辆意图集合(格式为JSON数组): ")
    task.set_vehicles_with_intentions(json.loads(vehicles_with_intentions_input))

    # 计算协作方集合
    coop_set = task.get_coope_set()
    # print(f"协作方集合: {coop_set}")

    # 计算关键车辆集合
    topological_vehicles = task.get_topological_set()
    # print(f"关键车辆集合: {topological_vehicles}")
